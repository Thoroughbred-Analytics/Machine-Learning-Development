import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def display_predictions(y_pred, X, y, names, num_predictions=10):
    """
    Displays a comparison of predicted and actual values.
    
    Args:
        y_pred:          Precomputed predictions (e.g. out-of-fold predictions from K-Fold)
        X:               Feature set (used only for length bounds)
        y:               Actual target values (pd.Series)
        names:           Horse names aligned to X/y index (pd.Series)
        num_predictions: Number of samples to display and plot
    """
    n = min(num_predictions, len(y_pred))

    # ── Console summary ──────────────────────────────────────────────────────
    print("Predicted vs Actual values:")
    num_correct = 0
    for i in range(n):
        predicted = y_pred[i]
        actual    = y.iloc[i]
        name      = names.iloc[i]
        if abs(predicted - actual) < 10:
            num_correct += 1
        print(f"{name}: Predicted = {predicted:.2f}, Actual = {actual:.2f}")

    accuracy = num_correct / n * 100
    print(f"\nAccuracy (within 10 rating points): {accuracy:.2f}%\n")

    # ── Build results dataframe ───────────────────────────────────────────────
    df_results = pd.DataFrame({
        'horse_name':       [names.iloc[i]  for i in range(n)],
        'actual_rating':    [y.iloc[i]      for i in range(n)],
        'predicted_rating': [y_pred[i]      for i in range(n)],
    })

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 8))
    x_pos = np.arange(n)

    ax.scatter(x_pos, df_results['actual_rating'],
               color='steelblue', s=120, label='Actual Rating',
               marker='o', alpha=0.8, zorder=3)

    ax.scatter(x_pos, df_results['predicted_rating'],
               color='orange', s=120, label='Predicted Rating',
               marker='s', alpha=0.8, zorder=3)

    for i in range(n):
        ax.plot([i, i],
                [df_results.iloc[i]['actual_rating'],
                 df_results.iloc[i]['predicted_rating']],
                color='gray', alpha=0.6, linewidth=2, zorder=1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_results['horse_name'], rotation=45, ha='right')
    ax.set_xlabel('Horse Names', fontsize=12)
    ax.set_ylabel('Rating', fontsize=12)
    ax.set_title('Horse Ratings: Model Predictions vs Actual Values', fontsize=14, pad=20)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, zorder=0)

    for i, row in df_results.iterrows():
        ax.annotate(f'{row["actual_rating"]:.2f}',
                    (i, row['actual_rating']),
                    xytext=(-15, 10), textcoords='offset points',
                    fontsize=9, color='steelblue', weight='bold', ha='center')

        ax.annotate(f'{row["predicted_rating"]:.2f}',
                    (i, row['predicted_rating']),
                    xytext=(15, 10), textcoords='offset points',
                    fontsize=9, color='orange', weight='bold', ha='center')

    plt.tight_layout()
    plt.show()