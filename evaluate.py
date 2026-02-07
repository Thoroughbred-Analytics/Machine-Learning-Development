import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def display_predictions(model, X_test, y_test, names_test, num_predictions=10):
    """
    Displays a comparison of predicted and actual values for the test set.
    
    Args:
        model: Trained model
        X_test: Test feature set
        y_test: Actual target values for the test set
        names_test: Names corresponding to the test set
        num_predictions: Number of predictions to display
    """
    y_pred = model.predict(X_test)

    # Print a short summary
    print("Predicted vs Actual values:")
    numCorrect = 0
    for i in range(min(num_predictions, len(y_pred))):
        if abs(y_pred[i] - y_test.iloc[i]) < 10:  # Consider it correct if within 10 rating points
            numCorrect += 1
        print(f"{names_test.iloc[i]}: Predicted = {y_pred[i]:.2f}, Actual = {y_test.iloc[i]:.2f}")
    accuracy = numCorrect / min(num_predictions, len(y_pred)) * 100
    print(f"\nAccuracy (within 10 rating points): {accuracy:.2f}%\n")


    # Prepare dataframe for plotting
    df_results = pd.DataFrame({
        'horse_name': list(names_test.iloc[:num_predictions]),
        'actual_rating': list(y_test.iloc[:num_predictions]),
        'predicted_rating': list(y_pred[:num_predictions])
    })

    fig, ax = plt.subplots(figsize=(14, 8))
    x_pos = np.arange(len(df_results))

    # plotting the actual ratings (blue circles)
    actual_plots = ax.scatter(x_pos, df_results['actual_rating'], 
                        color='steelblue', s=120, label='Actual Rating', 
                        marker='o', alpha=0.8, zorder=3)

    # plotting the predicted ratings (orange squares)
    predicted_plots = ax.scatter(x_pos, df_results['predicted_rating'], 
                           color='orange', s=120, label='Predicted Rating', 
                           marker='s', alpha=0.8, zorder=3)

    # connect the two plots with a vertical line
    for i in range(len(df_results)):
        ax.plot([i, i], 
                [df_results.iloc[i]['actual_rating'], df_results.iloc[i]['predicted_rating']], 
                color='gray', alpha=0.6, linewidth=2, zorder=1)
        
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_results['horse_name'], rotation=45, ha='right')
    ax.set_xlabel('Horse Names', fontsize=12)
    ax.set_ylabel('Rating', fontsize=12)    
    ax.set_title('Horse Ratings: Model Predictions vs Actual Values', fontsize=14,  pad=20)

    # Add legend
    ax.legend(loc='upper right', fontsize=11)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, zorder=0)

    # Add value labels next to dots
    for i, row in df_results.iterrows():
        ax.annotate(f'{row["actual_rating"]:.2f}', 
                    (i, row['actual_rating']), 
                    xytext=(-15, 10), textcoords='offset points',
                    fontsize=9, color='steelblue', weight='bold',
                    ha='center')

        ax.annotate(f'{row["predicted_rating"]:.2f}', 
                    (i, row['predicted_rating']), 
                    xytext=(15, 10), textcoords='offset points',
                    fontsize=9, color='orange', weight='bold',
                    ha='center')

    plt.tight_layout()
    plt.show()