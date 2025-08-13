 
import numpy as np
import editdistance


def calculate_accuracy(pred_str, true_str):
    """
    Character-level accuracy.
    """
    if len(true_str) == 0:
        return 0.0
    correct = sum(p == t for p, t in zip(pred_str, true_str))
    accuracy = correct / len(true_str)
    print(f"[ACCURACY] Pred: '{pred_str}' | True: '{true_str}' | Char Acc: {accuracy:.2f}")
    return accuracy


def calculate_wer(pred_str, true_str):
    """
    Word Error Rate (based on edit distance between words).
    """
    pred_words = pred_str.strip().split()
    true_words = true_str.strip().split()

    if not true_words:
        return 1.0 if pred_words else 0.0

    dist = editdistance.eval(pred_words, true_words)
    wer = dist / max(1, len(true_words))
    print(f"[WER] Pred: '{pred_str}' | True: '{true_str}' | WER: {wer:.2f}")
    return wer


def calculate_cer(pred_str, true_str):
    """
    Character Error Rate (edit distance between characters).
    """
    if not true_str:
        return 1.0 if pred_str else 0.0

    dist = editdistance.eval(pred_str, true_str)
    cer = dist / max(1, len(true_str))
    print(f"[CER] Pred: '{pred_str}' | True: '{true_str}' | CER: {cer:.2f}")
    return cer
