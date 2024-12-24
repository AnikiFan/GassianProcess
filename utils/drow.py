import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_pred(gts, preds, pred_cols="sklearn_pred", e_bar=None, title="stock_pred"):
    """
    绘制预测结果
    :param gts: 真实值 [HPQ, VZ, SBUX]
    :param preds: 预测值 [HPQ_test, VZ_test, SBUX_test]
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 9))
    for i, (gt, pred) in enumerate(zip(gts, preds)):
        gt = gt[gt["year"] == 2011]
        axes[i].plot(gt["year_day"], gt["year_adjClose"], label="True")
        axes[i].plot(pred["year_day"], pred[pred_cols], label=pred_cols)
        # if e_bar is not None:
        #     axes[i].fill_between(pred["year_day"], pred[pred_cols], 
        #                         pred[pred_cols] - 1.96 * pred[e_bar], pred[pred_cols] + 1.96 * pred[e_bar], 
        #                         alpha=0.3
        #                     )
        axes[i].set_title(["HPQ", "VZ", "SBUX"][i])
        axes[i].set_xlabel("year_day")
        axes[i].set_ylabel("adjClose")
        axes[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # 按照 4 个季度添加网络, 并在对应网格上方添加季度名称(Q1, Q2, Q3, Q4)
        for j in range(1, 251, 63):
            axes[i].axvline(j, color="gray", linestyle="--")
        axes[i].axhline(0, color="gray", linestyle="--")
        axes[i].text(63/2, axes[i].get_ylim()[1] - 0.5, "Q1", ha="center")
        axes[i].text(63 + 63/2, axes[i].get_ylim()[1] - 0.5, "Q2", ha="center")
        axes[i].text(2*63 + 63/2, axes[i].get_ylim()[1] - 0.5, "Q3", ha="center")
        axes[i].text(3*63 + 63/2, axes[i].get_ylim()[1] - 0.5, "Q4", ha="center")
        axes[i].grid(True, linestyle="-.")

    fig.suptitle(title)
    
    # 避免三个子图重叠
    plt.tight_layout()
    plt.savefig("figs/stock_pred.png", dpi=400)