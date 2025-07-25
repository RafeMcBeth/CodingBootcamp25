"""AI/ML Inference for Medical Physics Applications.

This script demonstrates how to use machine learning models for medical physics tasks.
We'll cover image classification, dose prediction, and quality assurance using AI.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import warnings
warnings.filterwarnings('ignore')


def generate_synthetic_organ_data():
    """Generate synthetic organ contour data for classification.
    
    Returns:
        Features (shape descriptors) and labels (organ types)
    """
    print("🧠 Generating synthetic organ data for AI training...")
    
    np.random.seed(42)  # For reproducible results
    n_samples_per_organ = 200
    
    # Define organ characteristics (area, perimeter, roundness, etc.)
    organs = {
        'Heart': {
            'area_mean': 150, 'area_std': 20,
            'perimeter_mean': 80, 'perimeter_std': 10,
            'roundness_mean': 0.7, 'roundness_std': 0.1,
            'eccentricity_mean': 0.8, 'eccentricity_std': 0.1
        },
        'Lung': {
            'area_mean': 300, 'area_std': 50,
            'perimeter_mean': 120, 'perimeter_std': 20,
            'roundness_mean': 0.4, 'roundness_std': 0.1,
            'eccentricity_mean': 0.9, 'eccentricity_std': 0.05
        },
        'Liver': {
            'area_mean': 400, 'area_std': 60,
            'perimeter_mean': 140, 'perimeter_std': 25,
            'roundness_mean': 0.5, 'roundness_std': 0.15,
            'eccentricity_mean': 0.6, 'eccentricity_std': 0.1
        },
        'Kidney': {
            'area_mean': 80, 'area_std': 15,
            'perimeter_mean': 50, 'perimeter_std': 8,
            'roundness_mean': 0.8, 'roundness_std': 0.1,
            'eccentricity_mean': 0.7, 'eccentricity_std': 0.1
        }
    }
    
    features = []
    labels = []
    
    for organ_name, params in organs.items():
        print(f"  Generating {n_samples_per_organ} {organ_name} contours...")
        
        for _ in range(n_samples_per_organ):
            # Generate features with some correlation
            area = np.random.normal(params['area_mean'], params['area_std'])
            perimeter = np.random.normal(params['perimeter_mean'], params['perimeter_std'])
            roundness = np.random.normal(params['roundness_mean'], params['roundness_std'])
            eccentricity = np.random.normal(params['eccentricity_mean'], params['eccentricity_std'])
            
            # Derived features
            compactness = 4 * np.pi * area / (perimeter ** 2)
            aspect_ratio = 1 / eccentricity if eccentricity > 0 else 1
            
            # Clip values to realistic ranges
            area = max(10, area)
            perimeter = max(10, perimeter)
            roundness = np.clip(roundness, 0.1, 1.0)
            eccentricity = np.clip(eccentricity, 0.1, 1.0)
            compactness = np.clip(compactness, 0.1, 1.0)
            
            features.append([area, perimeter, roundness, eccentricity, compactness, aspect_ratio])
            labels.append(organ_name)
    
    feature_names = ['Area', 'Perimeter', 'Roundness', 'Eccentricity', 'Compactness', 'Aspect_Ratio']
    
    print(f"✓ Generated {len(features)} organ samples with {len(feature_names)} features")
    
    return np.array(features), np.array(labels), feature_names


def train_organ_classifier(features, labels, feature_names):
    """Train a Random Forest classifier for organ identification.
    
    Args:
        features: Feature matrix
        labels: Organ labels
        feature_names: List of feature names
        
    Returns:
        Trained classifier and test results
    """
    print("\n🤖 TRAINING ORGAN CLASSIFIER")
    print("=" * 40)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train Random Forest classifier
    print("\nTraining Random Forest classifier...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Training completed!")
    print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Feature importance
    importances = classifier.feature_importances_
    
    print(f"\n📊 Feature Importance:")
    for name, importance in zip(feature_names, importances):
        print(f"  {name:12}: {importance:.3f}")
    
    # Detailed classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Feature importance plot
    plt.subplot(1, 3, 1)
    plt.bar(feature_names, importances)
    plt.title('Feature Importance')
    plt.xticks(rotation=45)
    plt.ylabel('Importance')
    
    # Confusion matrix visualization
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    organ_types = np.unique(labels)
    
    plt.subplot(1, 3, 2)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(organ_types))
    plt.xticks(tick_marks, organ_types, rotation=45)
    plt.yticks(tick_marks, organ_types)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add text annotations
    for i in range(len(organ_types)):
        for j in range(len(organ_types)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    
    # Sample predictions visualization
    plt.subplot(1, 3, 3)
    feature_idx = 0  # Area
    for i, organ in enumerate(organ_types):
        organ_mask = y_test == organ
        plt.scatter(X_test[organ_mask, feature_idx], 
                   X_test[organ_mask, 1],  # Perimeter
                   label=organ, alpha=0.7)
    
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Feature Space Visualization')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return classifier, accuracy


def generate_dose_prediction_data():
    """Generate synthetic dose prediction data.
    
    Returns:
        Treatment parameters and corresponding dose distributions
    """
    print("\n💊 Generating dose prediction training data...")
    
    np.random.seed(42)
    n_samples = 500
    
    # Treatment parameters
    beam_energy = np.random.choice([6, 10, 15, 18], n_samples)  # MV
    field_size = np.random.uniform(5, 20, n_samples)  # cm
    depth = np.random.uniform(1, 20, n_samples)  # cm
    angle = np.random.uniform(0, 360, n_samples)  # degrees
    
    # Synthetic dose calculation (simplified physics model)
    # PDD (Percent Depth Dose) calculation
    pdd_max_depth = 1.5 + 0.3 * beam_energy  # Depth of maximum dose
    mu_eff = 0.1 - 0.005 * beam_energy  # Effective attenuation coefficient
    
    pdd = np.zeros(n_samples)
    for i in range(n_samples):
        if depth[i] <= pdd_max_depth[i]:
            pdd[i] = (depth[i] / pdd_max_depth[i]) * 100
        else:
            pdd[i] = 100 * np.exp(-mu_eff[i] * (depth[i] - pdd_max_depth[i]))
    
    # Field size correction (output factor)
    output_factor = 0.8 + 0.02 * field_size
    
    # Combined dose calculation
    dose_rate = pdd * output_factor * (1 + 0.1 * np.random.normal(0, 1, n_samples))  # Add noise
    
    # Combine features
    features = np.column_stack([beam_energy, field_size, depth, angle])
    feature_names = ['Beam_Energy_MV', 'Field_Size_cm', 'Depth_cm', 'Gantry_Angle_deg']
    
    print(f"✓ Generated {n_samples} dose calculation samples")
    print(f"  Dose range: {dose_rate.min():.1f} to {dose_rate.max():.1f}%")
    
    return features, dose_rate, feature_names


def train_dose_predictor(features, dose_rates, feature_names):
    """Train a dose prediction model.
    
    Args:
        features: Treatment parameters
        dose_rates: Corresponding dose rates
        feature_names: Parameter names
        
    Returns:
        Trained regression model
    """
    print("\n🎯 TRAINING DOSE PREDICTOR")
    print("=" * 35)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, dose_rates, test_size=0.3, random_state=42
    )
    
    # Train Random Forest regressor
    print("Training dose prediction model...")
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)
    
    # Make predictions
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate percentage error
    percentage_error = np.abs((y_test - y_pred) / y_test) * 100
    mean_percentage_error = np.mean(percentage_error)
    
    print(f"✓ Training completed!")
    print(f"  RMSE: {rmse:.2f}%")
    print(f"  Mean percentage error: {mean_percentage_error:.2f}%")
    
    # Feature importance
    importances = regressor.feature_importances_
    print(f"\n📊 Feature Importance for Dose Prediction:")
    for name, importance in zip(feature_names, importances):
        print(f"  {name:18}: {importance:.3f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Predicted vs Actual
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Dose Rate (%)')
    plt.ylabel('Predicted Dose Rate (%)')
    plt.title('Dose Prediction Accuracy')
    
    # Residuals
    plt.subplot(1, 3, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Dose Rate (%)')
    plt.ylabel('Residuals (%)')
    plt.title('Prediction Residuals')
    
    # Feature importance
    plt.subplot(1, 3, 3)
    plt.bar(feature_names, importances)
    plt.title('Feature Importance')
    plt.xticks(rotation=45)
    plt.ylabel('Importance')
    
    plt.tight_layout()
    plt.show()
    
    return regressor, rmse


def demonstrate_qa_anomaly_detection():
    """Demonstrate AI-based quality assurance anomaly detection."""
    print("\n🔍 AI-BASED QA ANOMALY DETECTION")
    print("=" * 45)
    
    # Generate synthetic daily QA measurements
    np.random.seed(42)
    n_days = 100
    
    # Normal QA parameters
    dose_output = np.random.normal(100, 0.5, n_days)  # cGy, target = 100
    beam_flatness = np.random.normal(102, 1, n_days)  # %, target = 102
    beam_symmetry = np.random.normal(100, 0.8, n_days)  # %, target = 100
    
    # Introduce some anomalies
    anomaly_days = [20, 45, 75]
    dose_output[anomaly_days] += np.random.normal(0, 3, len(anomaly_days))
    beam_flatness[anomaly_days] += np.random.normal(0, 5, len(anomaly_days))
    beam_symmetry[anomaly_days] += np.random.normal(0, 4, len(anomaly_days))
    
    # Create feature matrix
    qa_features = np.column_stack([dose_output, beam_flatness, beam_symmetry])
    
    # Simple anomaly detection using statistical thresholds
    print("Performing anomaly detection...")
    
    # Calculate z-scores for each parameter
    z_scores = np.abs((qa_features - np.mean(qa_features, axis=0)) / np.std(qa_features, axis=0))
    
    # Flag as anomaly if any parameter has z-score > 2.5
    anomalies = np.any(z_scores > 2.5, axis=1)
    
    print(f"Detected {np.sum(anomalies)} anomalous days out of {n_days}")
    print(f"Anomaly detection rate: {np.sum(anomalies)/n_days*100:.1f}%")
    
    # Visualize QA trends and anomalies
    plt.figure(figsize=(15, 10))
    
    days = np.arange(1, n_days + 1)
    
    # Dose output
    plt.subplot(3, 1, 1)
    plt.plot(days, dose_output, 'b-', alpha=0.7, label='Daily measurements')
    plt.scatter(days[anomalies], dose_output[anomalies], color='red', s=50, 
                label='Detected anomalies', zorder=5)
    plt.axhline(y=100, color='g', linestyle='--', label='Target')
    plt.axhline(y=100+2, color='orange', linestyle=':', label='±2% tolerance')
    plt.axhline(y=100-2, color='orange', linestyle=':')
    plt.ylabel('Dose Output (cGy)')
    plt.title('Daily QA Monitoring with AI Anomaly Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Beam flatness
    plt.subplot(3, 1, 2)
    plt.plot(days, beam_flatness, 'g-', alpha=0.7)
    plt.scatter(days[anomalies], beam_flatness[anomalies], color='red', s=50, zorder=5)
    plt.axhline(y=102, color='g', linestyle='--')
    plt.axhline(y=102+3, color='orange', linestyle=':')
    plt.axhline(y=102-3, color='orange', linestyle=':')
    plt.ylabel('Beam Flatness (%)')
    plt.grid(True, alpha=0.3)
    
    # Beam symmetry
    plt.subplot(3, 1, 3)
    plt.plot(days, beam_symmetry, 'm-', alpha=0.7)
    plt.scatter(days[anomalies], beam_symmetry[anomalies], color='red', s=50, zorder=5)
    plt.axhline(y=100, color='g', linestyle='--')
    plt.axhline(y=100+3, color='orange', linestyle=':')
    plt.axhline(y=100-3, color='orange', linestyle=':')
    plt.ylabel('Beam Symmetry (%)')
    plt.xlabel('Day')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return anomalies, qa_features


def visualize_generated_data(organ_features, organ_labels, organ_feature_names, 
                           dose_features, dose_rates, dose_feature_names):
    """Visualize the characteristics of generated synthetic data.
    
    Args:
        organ_features: Organ classification features
        organ_labels: Organ type labels
        organ_feature_names: Names of organ features
        dose_features: Dose prediction features
        dose_rates: Dose rate targets
        dose_feature_names: Names of dose features
    """
    print("\n📊 VISUALIZING GENERATED DATA")
    print("=" * 40)
    
    # Create a comprehensive visualization
    plt.figure(figsize=(20, 12))
    
    # ===== ORGAN DATA VISUALIZATION =====
    organ_types = np.unique(organ_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(organ_types)))
    
    # Organ feature distributions
    plt.subplot(3, 4, 1)
    for i, organ in enumerate(organ_types):
        mask = organ_labels == organ
        plt.hist(organ_features[mask, 0], alpha=0.6, label=organ, color=colors[i], bins=20)
    plt.xlabel(organ_feature_names[0])
    plt.ylabel('Count')
    plt.title('Organ Area Distribution')
    plt.legend()
    
    plt.subplot(3, 4, 2)
    for i, organ in enumerate(organ_types):
        mask = organ_labels == organ
        plt.hist(organ_features[mask, 2], alpha=0.6, label=organ, color=colors[i], bins=20)
    plt.xlabel(organ_feature_names[2])
    plt.ylabel('Count')
    plt.title('Organ Roundness Distribution')
    plt.legend()
    
    # 2D scatter plot of organ features
    plt.subplot(3, 4, 3)
    for i, organ in enumerate(organ_types):
        mask = organ_labels == organ
        plt.scatter(organ_features[mask, 0], organ_features[mask, 1], 
                   alpha=0.6, label=organ, color=colors[i], s=20)
    plt.xlabel(organ_feature_names[0])
    plt.ylabel(organ_feature_names[1])
    plt.title('Organ Feature Space (Area vs Perimeter)')
    plt.legend()
    
    # Organ feature correlation heatmap
    plt.subplot(3, 4, 4)
    correlation_matrix = np.corrcoef(organ_features.T)
    im = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(organ_feature_names)), 
               [name[:8] for name in organ_feature_names], rotation=45)
    plt.yticks(range(len(organ_feature_names)), 
               [name[:8] for name in organ_feature_names])
    plt.title('Organ Feature Correlations')
    
    # ===== DOSE DATA VISUALIZATION =====
    # Dose parameter distributions
    plt.subplot(3, 4, 5)
    plt.hist(dose_features[:, 0], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel(dose_feature_names[0])
    plt.ylabel('Count')
    plt.title('Beam Energy Distribution')
    
    plt.subplot(3, 4, 6)
    plt.hist(dose_features[:, 1], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel(dose_feature_names[1])
    plt.ylabel('Count')
    plt.title('Field Size Distribution')
    
    plt.subplot(3, 4, 7)
    plt.hist(dose_features[:, 2], bins=20, alpha=0.7, color='salmon', edgecolor='black')
    plt.xlabel(dose_feature_names[2])
    plt.ylabel('Count')
    plt.title('Depth Distribution')
    
    plt.subplot(3, 4, 8)
    plt.hist(dose_rates, bins=30, alpha=0.7, color='gold', edgecolor='black')
    plt.xlabel('Dose Rate (%)')
    plt.ylabel('Count')
    plt.title('Dose Rate Distribution')
    
    # ===== DOSE RELATIONSHIPS =====
    # Show relationship between parameters and dose
    plt.subplot(3, 4, 9)
    scatter = plt.scatter(dose_features[:, 2], dose_rates, 
                         c=dose_features[:, 0], alpha=0.6, cmap='viridis')
    plt.colorbar(scatter, label='Beam Energy (MV)')
    plt.xlabel(dose_feature_names[2])
    plt.ylabel('Dose Rate (%)')
    plt.title('Dose vs Depth (colored by Energy)')
    
    plt.subplot(3, 4, 10)
    scatter = plt.scatter(dose_features[:, 1], dose_rates, 
                         c=dose_features[:, 0], alpha=0.6, cmap='plasma')
    plt.colorbar(scatter, label='Beam Energy (MV)')
    plt.xlabel(dose_feature_names[1])
    plt.ylabel('Dose Rate (%)')
    plt.title('Dose vs Field Size (colored by Energy)')
    
    # ===== SUMMARY STATISTICS =====
    plt.subplot(3, 4, 11)
    # Create a summary table visualization
    organ_counts = [np.sum(organ_labels == organ) for organ in organ_types]
    plt.bar(range(len(organ_types)), organ_counts, color=colors)
    plt.xticks(range(len(organ_types)), organ_types)
    plt.ylabel('Sample Count')
    plt.title('Organ Sample Distribution')
    
    plt.subplot(3, 4, 12)
    # Box plot of dose rates by beam energy
    energy_levels = np.unique(dose_features[:, 0])
    dose_by_energy = [dose_rates[dose_features[:, 0] == energy] for energy in energy_levels]
    plt.boxplot(dose_by_energy, labels=[f'{int(e)} MV' for e in energy_levels])
    plt.ylabel('Dose Rate (%)')
    plt.xlabel('Beam Energy')
    plt.title('Dose Distribution by Energy')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n📈 DATA SUMMARY STATISTICS")
    print("-" * 30)
    print(f"Organ Classification Dataset:")
    print(f"  Total samples: {len(organ_features)}")
    print(f"  Features: {len(organ_feature_names)}")
    print(f"  Organ types: {len(organ_types)}")
    for organ in organ_types:
        count = np.sum(organ_labels == organ)
        print(f"    {organ}: {count} samples")
    
    print(f"\nDose Prediction Dataset:")
    print(f"  Total samples: {len(dose_features)}")
    print(f"  Features: {len(dose_feature_names)}")
    print(f"  Dose range: {dose_rates.min():.1f} - {dose_rates.max():.1f}%")
    print(f"  Energy levels: {sorted(np.unique(dose_features[:, 0]))}")
    
    print(f"\n🎯 Feature Statistics:")
    print(f"Organ Features (mean ± std):")
    for i, name in enumerate(organ_feature_names):
        mean_val = np.mean(organ_features[:, i])
        std_val = np.std(organ_features[:, i])
        print(f"  {name:15}: {mean_val:7.2f} ± {std_val:5.2f}")
    
    print(f"\nDose Features (mean ± std):")
    for i, name in enumerate(dose_feature_names):
        mean_val = np.mean(dose_features[:, i])
        std_val = np.std(dose_features[:, i])
        print(f"  {name:18}: {mean_val:7.2f} ± {std_val:5.2f}")


def show_sample_data(organ_features, organ_labels, organ_feature_names,
                    dose_features, dose_rates, dose_feature_names):
    """Display sample training data to show what individual data points look like.
    
    Args:
        organ_features: Organ classification features
        organ_labels: Organ type labels  
        organ_feature_names: Names of organ features
        dose_features: Dose prediction features
        dose_rates: Dose rate targets
        dose_feature_names: Names of dose features
    """
    print("\n🔍 SAMPLE TRAINING DATA")
    print("=" * 50)
    
    # Show sample organ data
    print("🫁 ORGAN CLASSIFICATION SAMPLES")
    print("-" * 35)
    
    organ_types = np.unique(organ_labels)
    
    # Create a nice table header
    header = f"{'Organ Type':<10} | "
    for name in organ_feature_names:
        header += f"{name[:8]:<8} | "
    print(header)
    print("-" * len(header))
    
    # Show 3 samples per organ type
    for organ in organ_types:
        organ_indices = np.where(organ_labels == organ)[0]
        sample_indices = organ_indices[:3]  # First 3 samples
        
        for i, idx in enumerate(sample_indices):
            row = f"{organ:<10} | "
            for j, feature_val in enumerate(organ_features[idx]):
                row += f"{feature_val:8.2f} | "
            print(row)
        print()  # Empty line between organs
    
    # Show what these features mean
    print("📝 FEATURE MEANINGS:")
    print("  Area:        Size of organ contour (arbitrary units)")
    print("  Perimeter:   Boundary length of organ contour")  
    print("  Roundness:   How circular the shape is (0-1)")
    print("  Eccentricity: How elongated the shape is (0-1)")
    print("  Compactness: Ratio of area to perimeter squared")
    print("  Aspect_Ratio: Width to height ratio")
    
    # Show sample dose data  
    print(f"\n💊 DOSE PREDICTION SAMPLES")
    print("-" * 30)
    
    # Create dose table header
    dose_header = f"{'Sample':<8} | "
    for name in dose_feature_names:
        dose_header += f"{name[:12]:<12} | "
    dose_header += f"{'Dose_Rate':<10}"
    print(dose_header)
    print("-" * len(dose_header))
    
    # Show 10 diverse dose samples
    sample_indices = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    
    for i, idx in enumerate(sample_indices):
        if idx < len(dose_features):
            row = f"#{i+1:<7} | "
            for j, feature_val in enumerate(dose_features[idx]):
                if j == 0:  # Beam energy - show as integer
                    row += f"{int(feature_val):<12} | "
                elif j == 3:  # Angle - show as integer  
                    row += f"{int(feature_val):<12} | "
                else:  # Field size and depth - show with 1 decimal
                    row += f"{feature_val:<12.1f} | "
            row += f"{dose_rates[idx]:<10.1f}"
            print(row)
    
    print(f"\n📝 DOSE FEATURE MEANINGS:")
    print("  Beam_Energy_MV:     X-ray beam energy in megavolts")
    print("  Field_Size_cm:      Radiation field size in centimeters") 
    print("  Depth_cm:           Measurement depth in patient")
    print("  Gantry_Angle_deg:   Machine rotation angle")
    print("  Dose_Rate:          Calculated dose percentage")
    
    # Show some interesting patterns in the data
    print(f"\n🔬 INTERESTING DATA PATTERNS:")
    print("-" * 30)
    
    # Organ size patterns
    for organ in organ_types:
        mask = organ_labels == organ
        avg_area = np.mean(organ_features[mask, 0])  # Area is first feature
        avg_roundness = np.mean(organ_features[mask, 2])  # Roundness is third feature
        print(f"{organ:>6}: Avg area={avg_area:6.1f}, Avg roundness={avg_roundness:.2f}")
    
    print(f"\nDose patterns:")
    for energy in sorted(np.unique(dose_features[:, 0])):
        mask = dose_features[:, 0] == energy
        avg_dose = np.mean(dose_rates[mask])
        print(f"{int(energy):>3} MV: Average dose rate = {avg_dose:.1f}%")
    
    # Show train/test split example
    print(f"\n📊 TRAIN/TEST SPLIT PREVIEW:")
    print("-" * 25)
    
    from sklearn.model_selection import train_test_split
    
    # Show organ split
    X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(
        organ_features, organ_labels, test_size=0.3, random_state=42, stratify=organ_labels
    )
    
    print(f"Organ data split:")
    print(f"  Total samples: {len(organ_features)}")
    print(f"  Training: {len(X_train_org)} samples ({len(X_train_org)/len(organ_features)*100:.1f}%)")
    print(f"  Testing:  {len(X_test_org)} samples ({len(X_test_org)/len(organ_features)*100:.1f}%)")
    
    print(f"\nTraining set organ distribution:")
    for organ in organ_types:
        count = np.sum(y_train_org == organ)
        print(f"  {organ}: {count} samples")
    
    print(f"\nTesting set organ distribution:")  
    for organ in organ_types:
        count = np.sum(y_test_org == organ)
        print(f"  {organ}: {count} samples")
    
    # Show dose split
    X_train_dose, X_test_dose, y_train_dose, y_test_dose = train_test_split(
        dose_features, dose_rates, test_size=0.3, random_state=42
    )
    
    print(f"\nDose data split:")
    print(f"  Total samples: {len(dose_features)}")
    print(f"  Training: {len(X_train_dose)} samples")
    print(f"  Testing:  {len(X_test_dose)} samples")
    print(f"  Training dose range: {y_train_dose.min():.1f} - {y_train_dose.max():.1f}%")
    print(f"  Testing dose range:  {y_test_dose.min():.1f} - {y_test_dose.max():.1f}%")


def main():
    """Main function demonstrating AI applications in medical physics."""
    print("🤖 AI/ML Applications in Medical Physics")
    print("=" * 50)
    print("Learn how machine learning can enhance medical physics workflows!")
    
    # Part 1: Generate and visualize organ data
    features, labels, feature_names = generate_synthetic_organ_data()
    
    # Part 2: Generate dose prediction data
    dose_features, dose_rates, dose_feature_names = generate_dose_prediction_data()
    
    # NEW: Show sample training data  
    show_sample_data(features, labels, feature_names,
                    dose_features, dose_rates, dose_feature_names)
    
    # Part 3: Visualize the generated data
    visualize_generated_data(features, labels, feature_names, 
                           dose_features, dose_rates, dose_feature_names)
    
    # Part 3: Train models
    classifier, accuracy = train_organ_classifier(features, labels, feature_names)
    
    # Part 4: Train dose predictor
    regressor, rmse = train_dose_predictor(dose_features, dose_rates, dose_feature_names)
    
    # Part 5: QA Anomaly Detection
    anomalies, qa_data = demonstrate_qa_anomaly_detection()
    
    # Interactive demonstration
    print("\n" + "="*50)
    print("🎯 INTERACTIVE AI DEMONSTRATION")
    print("="*50)
    
    # Test organ classification
    print("\n🔬 Testing organ classifier with new sample:")
    test_sample = np.array([[180, 85, 0.65, 0.75, 0.8, 1.3]])  # Heart-like features
    prediction = classifier.predict(test_sample)[0]
    probabilities = classifier.predict_proba(test_sample)[0]
    
    print(f"Sample features: Area=180, Perimeter=85, Roundness=0.65")
    print(f"Predicted organ: {prediction}")
    print("Prediction probabilities:")
    for organ, prob in zip(classifier.classes_, probabilities):
        print(f"  {organ}: {prob:.3f}")
    
    # Test dose prediction
    print("\n💊 Testing dose predictor with new treatment:")
    test_treatment = np.array([[10, 15, 5, 180]])  # 10MV, 15x15cm, 5cm depth, 180° angle
    predicted_dose = regressor.predict(test_treatment)[0]
    
    print(f"Treatment: 10 MV, 15×15 cm field, 5 cm depth, 180° gantry")
    print(f"Predicted dose rate: {predicted_dose:.1f}%")
    
    # Summary
    print("\n" + "="*50)
    print("🎓 AI LEARNING SUMMARY")
    print("="*50)
    print(f"✓ Trained organ classifier with {accuracy:.1%} accuracy")
    print(f"✓ Built dose predictor with {rmse:.1f}% RMSE")
    print(f"✓ Implemented QA anomaly detection")
    print(f"✓ Demonstrated real-time AI inference")
    
    print("\n💡 Key AI Concepts Learned:")
    print("  • Supervised learning (classification & regression)")
    print("  • Feature engineering for medical data")
    print("  • Model evaluation and validation")
    print("  • Anomaly detection for quality assurance")
    print("  • Feature importance analysis")
    
    print("\n🚀 Next Steps:")
    print("  • Try deep learning with neural networks")
    print("  • Implement image-based AI models")
    print("  • Explore automated treatment planning")
    print("  • Build real-time prediction systems")
    
    print("\n🔧 Required packages: numpy, matplotlib, scikit-learn")
    print("   Install with: pip install -r requirements.txt")
    print("   Or individually: pip install numpy matplotlib scikit-learn")


if __name__ == "__main__":
    main() 