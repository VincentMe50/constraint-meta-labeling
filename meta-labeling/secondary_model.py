import numpy as np 
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
import torch.optim as optim



class RegressionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1=32, hidden_dim_2=16, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(  
            nn.Linear(input_dim, hidden_dim_1), # Increased layer size, tune this 
            nn.ReLU(),  
            nn.BatchNorm1d(hidden_dim_1), # Added BatchNorm
            nn.Dropout(dropout_rate), # Adjusted, tune this 
            nn.Linear(hidden_dim_1, hidden_dim_2),  
            nn.ReLU(),  
            nn.BatchNorm1d(hidden_dim_2), # Added BatchNorm
            nn.Dropout(dropout_rate), # Adjusted, tune this
            nn.Linear(hidden_dim_2, 1)  
        )

    def forward(self, x):
        return self.net(x)

def custom_loss_nn(outputs, targets, model_nn_params, lambda_l2_custom):
    mse_loss = nn.MSELoss()(outputs, targets)
    # If using weight_decay in Adam, this manual L2 might be redundant or need coordination
    l2_reg = sum(p.pow(2).sum() for p in model_nn_params if p.requires_grad)
    return mse_loss + lambda_l2_custom * l2_reg

def secondary_model(
    asset_idx,
    X_test_features_from_primary, # These are the original features for the test set period
    primary_model_targets_on_test_set, # These are y_test_pm (e.g. label_mat[n_train:, asset_idx])
    primary_model_predictions_on_test_set, # These are y_pred_primary_on_test
    primary_signals_to_refine, # This is y_pred_primary_on_test, passed for modification
    #fixed_params_nn,
    #param_grid_nn,    
    # scaler_pm, # If using same scaled features
    # TODO: Tune these hyperparameters
    nn_input_dim,
    threshold_meta_adjustment=0.05, # When to apply the meta-model's correction
    lambda_l2_custom_nn=1e-5, 
    lr_nn=0.001,
    weight_decay_nn=0.0001,  # Adam's L2, distinct from custom_loss_nn's lambda_l2
    pre_tuned_params_nn=None,
    ):

    # 1. Define Meta-Labels: Error of the primary model
    meta_labels_true_error = primary_model_targets_on_test_set - primary_model_predictions_on_test_set
    
    # 2. Input Features for Secondary Model:
    # Option A: Use original features (X_test_features_from_primary)
    # Option B: Use primary model's predictions as features (or part of features)
    # Option C(this time): Combine original features and primary model's predictions/confidence
    secondary_model_input_features = np.concatenate(
        (X_test_features_from_primary, primary_model_predictions_on_test_set.reshape(-1, 1)), axis=1
    )
    current_nn_input_dim = secondary_model_input_features.shape[1]

    # Prepare data for PyTorch
    X_meta_tensor = torch.tensor(secondary_model_input_features, dtype=torch.float32)
    meta_labels_error_tensor = torch.tensor(meta_labels_true_error, dtype=torch.float32).view(-1, 1) 

    X_train_meta, X_test_meta, y_train_meta, y_test_meta = train_test_split(
        X_meta_tensor, meta_labels_error_tensor, test_size=0.3, random_state=42, shuffle=False # No shuffle for time series generally [cite: 6]
    )
    # print("y_test_meta shape (secondary model):", y_test_meta.shape)


    # 3. Define the secondary model
    model_nn = RegressionNN(input_dim=current_nn_input_dim)
    optimizer_nn = optim.Adam(model_nn.parameters(), lr=lr_nn, weight_decay=weight_decay_nn)
    
    num_epochs_nn = 200 # Reduced for example, tune this
    batch_size_nn = 16 # Reduced for example, tune this
    # train_losses_nn = [] 
    
    for epoch in range(num_epochs_nn):
        model_nn.train()
        permutation = torch.randperm(X_train_meta.size(0))
        for i in range(0, X_train_meta.size(0), batch_size_nn):
            indices = permutation[i:i+batch_size_nn]
            batch_X_meta = X_train_meta[indices]
            batch_y_meta = y_train_meta[indices]

            outputs_meta = model_nn(batch_X_meta)
            # loss = custom_loss_nn(outputs_meta, batch_y_meta, model_nn.parameters(), lambda_l2_custom_nn)
            loss = nn.MSELoss()(outputs_meta, batch_y_meta)

            optimizer_nn.zero_grad()
            loss.backward()
            optimizer_nn.step()

        # Log training loss
        # with torch.no_grad():
        #     model_nn.eval()
        #     train_pred_meta = model_nn(X_train_meta)
        #     current_train_loss = nn.MSELoss()(train_pred_meta, y_train_meta).item()
        #     train_losses_nn.append(current_train_loss)
        #     if (epoch + 1) % 50 == 0: # [cite: 10]
        #         print(f'Asset {asset_idx}, NN Epoch [{epoch+1}/{num_epochs_nn}], Train Loss: {current_train_loss:.4f}')
    
    # Evaluate secondary model on its test set
    model_nn.eval()
    with torch.no_grad():
        test_pred_meta_errors = model_nn(X_test_meta)
        final_test_loss_meta = nn.MSELoss()(test_pred_meta_errors, y_test_meta).item()
        print(f'\nAsset {asset_idx}, NN Final Test Loss (predicting error): {final_test_loss_meta:.4f}')

    # Predict errors for the entire primary model's test set (for refinement)
    with torch.no_grad():
        all_pred_errors_meta = model_nn(X_meta_tensor).numpy().flatten()
    

    # Refine primary signals
    # Adjustment logic: signal = original_signal - predicted_error
    refined_signals = primary_signals_to_refine.copy()
    
    adjustment_mask = np.abs(all_pred_errors_meta) > threshold_meta_adjustment
    
    refined_signals[adjustment_mask] = primary_signals_to_refine[adjustment_mask] - all_pred_errors_meta[adjustment_mask]
    
    refined_signals = np.clip(refined_signals, 0, 1) # Assuming signals should be between 0 and 1
    
    print(f"Asset {asset_idx}: Number of signals refined by meta-model: {np.sum(adjustment_mask)}\n")
    return refined_signals

