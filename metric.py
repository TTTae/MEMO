import torch


def get_recall(scores, target): #recall --> wether next item in session is within top K=20 recommended items or not
    """
    Calculates the recall score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
    """
    target = target.data.cpu()
    scores = scores.data.cpu()
    # top 10
    val, idxx = scores.topk(10, 1)
    target_new = target.view(-1, 1).expand_as(idxx)
    hits = (target_new == idxx).nonzero(as_tuple=False)
    if len(hits) == 0:
        return 0
    n_hits = (target_new == idxx).nonzero(as_tuple=False)[:, :-1].size(0)
    recall = float(n_hits)
    return recall


def get_mrr(scores, target): #Mean Receiprocal Rank --> Average of rank of next item in the session.
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        mrr (float): the mrr score
    """

    target = target.data.cpu()
    scores = scores.data.cpu()
    # top 10
    val, idxx = scores.data.topk(10, 1)

    target_new = target.view(-1, 1).expand_as(idxx)
    hits = (target_new == idxx).nonzero(as_tuple=False)
    if len(hits) == 0:
        return 0
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data.cpu().numpy()
    return mrr


def evaluate(scores, target):
    """
    Evaluates the model using Recall@K, MRR@K scores.

    Args:
        logits (B,C): torch.LongTensor. The predicted logit for the next items.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
    """

    recall = get_recall(scores, target)
    mrr = get_mrr(scores, target)
    return recall, mrr
