import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmts_utils.shift import compute_cluster_based_shift
from mmts_utils.temperature import compute_tau_base

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

def gather_features(
    *feature_tensors,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
):
    assert has_distributed, \
        "torch.distributed did not import correctly, please use a PyTorch version with support."

    if use_horovod:
        assert hvd is not None, "Please install horovod"
        if gather_with_grad:
            all_features = [hvd.allgather(f) for f in feature_tensors]
        else:
            with torch.no_grad():
                all_features = [hvd.allgather(f) for f in feature_tensors]

            if not local_loss:
                all_features = [
                    torch.cat(
                        [*chunks[:rank], original, *chunks[rank + 1:]],
                        dim=0,
                    )
                    for original, gathered in zip(feature_tensors, all_features)
                    for chunks in [list(gathered.chunk(world_size, dim=0))]
                ]
    else:
        if gather_with_grad:
            all_features = [
                torch.cat(torch.distributed.nn.all_gather(f), dim=0)
                for f in feature_tensors
            ]
        else:
            all_features = []
            for original in feature_tensors:
                buckets = [torch.zeros_like(original) for _ in range(world_size)]
                dist.all_gather(buckets, original)
                
                if not local_loss:
                    buckets[rank] = original
                
                all_features.append(torch.cat(buckets, dim=0))

    return all_features

class ClipLoss(nn.Module):

    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"loss": total_loss}


class MMTSClipLoss(nn.Module):

    def __init__(
        self,
        alpha,
        number_steps_per_period,
        distribution_path,
        min_shift,
        max_shift,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        self.alpha = alpha
        self.number_steps_per_period = number_steps_per_period

        self.dist_df = pd.read_csv(distribution_path)
        self.min_cluster_size = int(self.dist_df["count"].min())
        self.max_cluster_size = int(self.dist_df["count"].max())

        self.min_shift = min_shift
        self.max_shift = max_shift

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def set_number_steps_per_period(self, number_steps_per_period: int):
        self.number_steps_per_period = number_steps_per_period

    def _compute_tau(self, global_step: int, clusters: torch.Tensor) -> float:
        base_tau = compute_tau_base(
            global_step=global_step, alpha=self.alpha, number_steps_per_period=self.number_steps_per_period
        )

        per_sample_shift = compute_cluster_based_shift(
            clusters=clusters,
            min_cluster_size=self.min_cluster_size,
            max_cluster_size=self.max_cluster_size,
            min_shift=self.min_shift,
            max_shift=self.max_shift,
        )

        return base_tau + per_sample_shift

    def forward(self, image_features, text_features, global_step, clusters):
        device = image_features.device

        if self.world_size > 1:
            all_image_features, all_text_features, all_clusters = gather_features(
                image_features,
                text_features,
                clusters,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                tau = self._compute_tau(global_step, clusters)
                logit_scale = (1.0 / tau).unsqueeze(1)
                logits_per_image = logit_scale * (image_features @ all_text_features.T)
                logits_per_text = logit_scale * (text_features @ all_image_features.T)
            else:
                tau = self._compute_tau(global_step, all_clusters)
                logit_scale = (1.0 / tau).unsqueeze(1)
                logits_per_image = (
                    logit_scale * (all_image_features @ all_text_features.T)
                )
                logits_per_text =  logit_scale * (all_text_features @ all_image_features.T)
        else:
            tau = self._compute_tau(global_step, clusters)
            logit_scale = (1.0 / tau).unsqueeze(1)

            logits_per_image = logit_scale * (image_features @ text_features.T)
            logits_per_text = logit_scale * (text_features @ image_features.T)

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"loss": total_loss}


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class MaxMarginRankingLoss(nn.Module):

    def __init__(
        self,
        margin=0.2,
        fix_norm=True,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.fix_norm = fix_norm
        self.margin = margin
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

    def forward(self, image_features, text_features, weight=None):
        # TODO: try gather_from_all in
        # https://github.com/facebookresearch/LaViLa/blob/main/lavila/models/distributed_utils.py
        # all_image_features = gather_from_all(image_features)
        # all_text_features = gather_from_all(text_features)
        all_image_features, all_text_features = gather_features(
            image_features,
            text_features,
            local_loss=self.local_loss,
            gather_with_grad=self.gather_with_grad,
            rank=self.rank,
            world_size=self.world_size,
            use_horovod=self.use_horovod,
        )

        x = sim_matrix(all_text_features, all_image_features)

        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return {"loss": max_margin.mean()}


class MMTSMaxMarginRankingLoss(nn.Module):

    def __init__(
        self,
        alpha,
        number_steps_per_period,
        distribution_path,
        min_shift,
        max_shift,
        fix_norm=True,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.fix_norm = fix_norm
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        self.alpha = alpha
        self.number_steps_per_period = number_steps_per_period

        self.dist_df = pd.read_csv(distribution_path)
        self.min_cluster_size = int(self.dist_df["count"].min())
        self.max_cluster_size = int(self.dist_df["count"].max())
        
        self.min_shift = min_shift
        self.max_shift = max_shift

    def _compute_tau(self, global_step: int, clusters: torch.Tensor) -> float:
        base_tau = compute_tau_base(
            global_step=global_step, alpha=self.alpha, number_steps_per_period=self.number_steps_per_period
        )

        per_sample_shift = compute_cluster_based_shift(
            clusters=clusters,
            min_cluster_size=self.min_cluster_size,
            max_cluster_size=self.max_cluster_size,
            min_shift=self.min_shift,
            max_shift=self.max_shift,
        )

        return base_tau + per_sample_shift

    def forward(self, image_features, text_features, global_step, clusters):
        # TODO: try gather_from_all in
        # https://github.com/facebookresearch/LaViLa/blob/main/lavila/models/distributed_utils.py
        # all_image_features = gather_from_all(image_features)
        # all_text_features = gather_from_all(text_features)
        # all_clusters = gather_from_all(clusters)
        all_image_features, all_text_features, all_clusters = gather_features(
            image_features,
            text_features,
            clusters,
            local_loss=self.local_loss,
            gather_with_grad=self.gather_with_grad,
            rank=self.rank,
            world_size=self.world_size,
            use_horovod=self.use_horovod,
        )

        x = sim_matrix(all_text_features, all_image_features)

        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)

        # Compute margin based on MMTS logic
        per_sample_margin = self._compute_tau(global_step, all_clusters)
        per_sample_margin_expanded = per_sample_margin.repeat_interleave(n)
        per_sample_margin_expanded = torch.cat([per_sample_margin_expanded, per_sample_margin_expanded], 0).unsqueeze(1)

        max_margin = F.relu(per_sample_margin_expanded - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()

            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)

            per_sample_margin_ = torch.index_select(per_sample_margin_expanded, dim=0, index=keep_idx)
            max_margin = F.relu(per_sample_margin_ - (x1_ - x2_))

        return {"loss": max_margin.mean()}