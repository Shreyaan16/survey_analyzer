import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import io

# Set page configuration
st.set_page_config(
    page_title="Doctor Availability Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and preprocess the dataset with advanced caching
@st.cache_data
def load_data(file_path='dummy_npi_data.xlsx'):
    """Load and preprocess the NPI dataset with comprehensive feature engineering"""
    try:
        data = pd.read_excel(file_path)
        
        # Convert datetime columns
        data['Login Time'] = pd.to_datetime(data['Login Time'])
        data['Logout Time'] = pd.to_datetime(data['Logout Time'])
        
        # Extract basic time features
        data['Login Hour'] = data['Login Time'].dt.hour
        data['Login Minute'] = data['Login Time'].dt.minute
        data['Logout Hour'] = data['Logout Time'].dt.hour
        data['Logout Minute'] = data['Logout Time'].dt.minute
        data['Login Day'] = data['Login Time'].dt.day_of_week
        data['Session Duration'] = (data['Logout Time'] - data['Login Time']).dt.total_seconds() / 60
        
        # Verify session duration matches Usage Time
        if not np.allclose(data['Session Duration'], data['Usage Time (mins)'], rtol=1e-2):
            st.warning("Calculated session duration doesn't match 'Usage Time (mins)'. Using provided values.")
        
        # Create additional features
        data['Survey Engagement Ratio'] = data['Count of Survey Attempts'] / data['Usage Time (mins)']
        data['Survey Engagement Ratio'] = data['Survey Engagement Ratio'].fillna(0)
        
        # Time of day features (morning, afternoon, evening, night)
        data['Time of Day'] = pd.cut(
            data['Login Hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            right=False
        )
        
        # Encode categorical variables with label encoders
        categorical_columns = ['State', 'Region', 'Speciality', 'Time of Day']
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            data[f'{col}_encoded'] = le.fit_transform(data[col])
            label_encoders[col] = le
            
        # Create normalized survey engagement metric
        data['Normalized Survey Engagement'] = (data['Count of Survey Attempts'] / data['Count of Survey Attempts'].max()) * \
                                              (data['Usage Time (mins)'] / data['Usage Time (mins)'].max())
        
        # Create hour activity indicators (24 binary columns)
        for hour in range(24):
            data[f'Active_Hour_{hour}'] = data.apply(
                lambda row: is_hour_in_session(row['Login Hour'], row['Login Minute'], 
                                               row['Logout Hour'], row['Logout Minute'], hour),
                axis=1
            )
        
        return data, label_encoders
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def is_hour_in_session(login_hour, login_minute, logout_hour, logout_minute, target_hour):
    """Check if a specific hour is within a session timeframe"""
    login_minutes = login_hour * 60 + login_minute
    logout_minutes = logout_hour * 60 + logout_minute
    target_minutes_start = target_hour * 60
    target_minutes_end = (target_hour + 1) * 60
    
    # Handle sessions that cross midnight
    if logout_minutes < login_minutes:
        return (login_minutes <= target_minutes_start < 24*60) or (0 <= target_minutes_start < logout_minutes)
    
    # Normal case
    return login_minutes <= target_minutes_start < logout_minutes

# Function to check if a doctor is active at a given time with improved accuracy
def is_active_at_time(row, target_time):
    """Determine if a doctor is active at the specified time"""
    target_hour = target_time.hour
    target_minute = target_time.minute
    login_hour = row['Login Hour']
    login_minute = row['Login Minute']
    logout_hour = row['Logout Hour']
    logout_minute = row['Logout Minute']

    target_minutes = target_hour * 60 + target_minute
    login_minutes = login_hour * 60 + login_minute
    logout_minutes = logout_hour * 60 + logout_minute

    # Handle sessions that cross midnight
    if logout_minutes < login_minutes:  
        return (login_minutes <= target_minutes < 24*60) or (0 <= target_minutes < logout_minutes)
    
    return login_minutes <= target_minutes < logout_minutes

# Train the model with Random Forest only
@st.cache_resource
def train_model(data, feature_subset=None):
    """Train a Random Forest model to predict doctor availability"""
    if feature_subset is None:
        feature_subset = [
            'Login Hour', 'Login Minute', 'Logout Hour', 'Logout Minute', 
            'Usage Time (mins)', 'State_encoded', 'Region_encoded', 
            'Speciality_encoded', 'Normalized Survey Engagement'
        ]
    
    X = data[feature_subset]
    y = data['Active_at_Target']
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ))
    ])
    
    pipeline.fit(X, y)
    
    importances = pipeline.named_steps['model'].feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_subset,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return pipeline, feature_importance

# Function to evaluate model performance
def evaluate_model(model, X, y):
    """Evaluate model using cross-validation and precision-recall metrics"""
    metrics = {}
    
    if len(np.unique(y)) > 1:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc', error_score='raise')
        metrics['cv_scores'] = cv_scores
        metrics['mean_cv_score'] = cv_scores.mean()
        
        y_scores = model.predict_proba(X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y, y_scores)
        avg_precision = average_precision_score(y, y_scores)
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['thresholds'] = thresholds
        metrics['avg_precision'] = avg_precision
    else:
        metrics['cv_scores'] = None
        metrics['mean_cv_score'] = None
        metrics['precision'] = None
        metrics['recall'] = None
        metrics['thresholds'] = None
        metrics['avg_precision'] = None
    
    return metrics

# Function to generate interactive plots
def generate_plots(available_doctors, data):
    """Generate interactive plots using Plotly"""
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution by State", "Distribution by Region", 
                                     "Distribution by Specialty", "Time Analysis"])
    
    with tab1:
        state_counts = available_doctors['State'].value_counts().reset_index()
        state_counts.columns = ['State', 'Count']
        fig_state = px.bar(
            state_counts, 
            x='State', 
            y='Count', 
            color='Count',
            title="Doctors Available by State",
            color_continuous_scale=px.colors.sequential.Blues
        )
        fig_state.update_layout(xaxis_title="State", yaxis_title="Number of Doctors")
        st.plotly_chart(fig_state, use_container_width=True)
    
    with tab2:
        region_counts = available_doctors['Region'].value_counts().reset_index()
        region_counts.columns = ['Region', 'Count']
        fig_region = px.pie(
            region_counts, 
            values='Count', 
            names='Region',
            title="Regional Distribution of Available Doctors",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_region.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_region, use_container_width=True)
    
    with tab3:
        specialty_counts = available_doctors['Speciality'].value_counts().reset_index()
        specialty_counts.columns = ['Speciality', 'Count']
        fig_specialty = px.bar(
            specialty_counts, 
            x='Speciality', 
            y='Count', 
            title="Doctors Available by Specialty",
            color='Count',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        fig_specialty.update_layout(xaxis_title="Specialty", yaxis_title="Number of Doctors")
        st.plotly_chart(fig_specialty, use_container_width=True)
    
    with tab4:
        hour_cols = [f'Active_Hour_{hour}' for hour in range(24)]
        hourly_activity = data[hour_cols].sum().reset_index()
        hourly_activity.columns = ['Hour', 'Active Count']
        hourly_activity['Hour'] = hourly_activity['Hour'].str.replace('Active_Hour_', '').astype(int)
        fig_time = px.line(
            hourly_activity, 
            x='Hour', 
            y='Active Count',
            title="Doctor Activity Pattern by Hour of Day",
            markers=True
        )
        fig_time.update_layout(
            xaxis_title="Hour of Day (24-hour format)",
            yaxis_title="Number of Active Doctors",
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
        )
        st.plotly_chart(fig_time, use_container_width=True)

# New function to calculate peak availability times
def get_peak_availability_times(data):
    """Calculate the top 5 hours with the most active doctors"""
    hour_cols = [f'Active_Hour_{hour}' for hour in range(24)]
    hourly_activity = data[hour_cols].sum().reset_index()
    hourly_activity.columns = ['Hour', 'Active Count']
    hourly_activity['Hour'] = hourly_activity['Hour'].str.replace('Active_Hour_', '').astype(int)
    
    # Convert to 24-hour time format (e.g., 0 → "00:00", 13 → "13:00")
    hourly_activity['Time'] = hourly_activity['Hour'].apply(lambda x: f"{x:02d}:00")
    
    # Sort by active count and get top 5
    peak_times = hourly_activity.sort_values(by='Active Count', ascending=False).head(5)
    return peak_times

# Streamlit app with improved UI and functionality
def main():
    st.sidebar.image("https://img.icons8.com/color/96/000000/doctor-male--v1.png", width=100)
    st.sidebar.title("Configuration")
    
    uploaded_file = st.sidebar.file_uploader("Upload NPI data (Excel Format)", type=["xlsx", "xls"])
    if uploaded_file is not None:
        file_path = uploaded_file
    else:
        file_path = 'dummy_npi_data.xlsx'
    
    data, label_encoders = load_data(file_path)
    if data is None:
        st.error("Failed to load data. Please check the file format and try again.")
        return
    
    with st.sidebar.expander("Advanced Options"):
        likelihood_threshold = st.slider(
            "Likelihood Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.05,
            help="Minimum probability threshold for considering a doctor available"
        )
        show_model_details = st.checkbox(
            "Show Model Details", 
            value=False,
            help="Display model performance metrics and feature importance"
        )
    
    st.title("🩺 Doctor Availability Predictor")
    st.write("""
    This app predicts which doctors are most likely to be available for surveys at a specific time using a Random Forest model.
    Enter a time below to see the results and download the list of available doctors.
    """)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        time_input = st.text_input(
            "Enter time (24-hour format, e.g., '17:50')", 
            value="17:50",
            help="Use 24-hour format (HH:MM)"
        )
    with col2:
        st.write("") # Spacer
        st.write("") # Spacer
        analyze_button = st.button("Analyze", type="primary")
    
    if not analyze_button and not st.session_state.get('analyzed', False):
        st.info("Enter a time and click 'Analyze' to see predictions")
        return
    
    st.session_state['analyzed'] = True
    
    try:
        target_time = datetime.strptime(time_input, '%H:%M')
    except ValueError:
        st.error("Invalid time format. Please use 'HH:MM' (e.g., '17:50').")
        return
    
    with st.spinner("Analyzing doctor availability..."):
        # Set target variable for the given time
        data['Active_at_Target'] = data.apply(lambda row: is_active_at_time(row, target_time), axis=1)
        
        # Check if there are two classes
        has_two_classes = len(np.unique(data['Active_at_Target'])) > 1
        
        if has_two_classes:
            # Features for Random Forest
            features = [
                'Login Hour', 'Login Minute', 'Logout Hour', 'Logout Minute', 
                'Usage Time (mins)', 'State_encoded', 'Region_encoded', 
                'Speciality_encoded', 'Normalized Survey Engagement'
            ]
            
            # Train or load the model
            model, feature_importance = train_model(data, features)
            
            # Predict likelihood
            X = data[features]
            data['Likelihood'] = model.predict_proba(X)[:, 1]
            
            # Evaluate model performance
            if show_model_details:
                eval_metrics = evaluate_model(model, X, data['Active_at_Target'])
        else:
            # If only one class, skip model training and use Active_at_Target as Likelihood
            data['Likelihood'] = data['Active_at_Target'].astype(float)
            feature_importance = None
            eval_metrics = None
            st.warning(f"Only one class detected at {time_input}. No predictive modeling performed; showing actual activity status.")
        
        # Filter and sort doctors
        all_doctors = data[['NPI', 'State', 'Region', 'Speciality', 'Active_at_Target', 'Likelihood']]
        active_doctors = all_doctors[all_doctors['Active_at_Target'] == 1].copy()
        likely_doctors = all_doctors[all_doctors['Likelihood'] >= likelihood_threshold].copy()
        
        available_doctors = pd.concat([active_doctors, likely_doctors]).drop_duplicates().sort_values(by='Likelihood', ascending=False)
        available_doctors['Likelihood (%)'] = (available_doctors['Likelihood'] * 100).round(1)
    
    st.header(f"📊 Doctors Available at {time_input}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Available Doctors", len(available_doctors))
    with col2:
        st.metric("Active Now", len(active_doctors))
    with col3:
        st.metric("Likely to Respond", len(likely_doctors))
    
    if not available_doctors.empty:
        display_df = available_doctors[['NPI', 'State', 'Region', 'Speciality', 'Likelihood (%)', 'Active_at_Target']]
        display_df = display_df.rename(columns={'Active_at_Target': 'Currently Active'})
        
        st.dataframe(
            display_df,
            column_config={
                "NPI": st.column_config.TextColumn("NPI"),
                "Likelihood (%)": st.column_config.ProgressColumn(
                    "Response Likelihood",
                    format="%f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Currently Active": st.column_config.CheckboxColumn(
                    "Currently Active",
                    help="Whether the doctor is currently logged in",
                ),
            },
            hide_index=True
        )
        
        # Dropdown for download format selection
        download_format = st.selectbox(
            "Select download format",
            options=["CSV", "Excel"],
            key="download_doctors_format"
        )
        
        if download_format == "CSV":
            csv = available_doctors.to_csv(index=False)
            st.download_button(
                label="Download Doctors Data",
                data=csv,
                file_name=f"doctors_at_{time_input.replace(':', '_')}.csv",
                mime="text/csv"
            )
        else:  # Excel
            excel_buffer = io.BytesIO()
            available_doctors.to_excel(excel_buffer, index=False)
            excel_data = excel_buffer.getvalue()
            st.download_button(
                label="Download Doctors Data",
                data=excel_data,
                file_name=f"doctors_at_{time_input.replace(':', '_')}.xlsx",
                mime="application/vnd.ms-excel"
            )
    else:
        st.warning("No doctors are predicted to be available at this time.")
    
    if not available_doctors.empty:
        st.header("📈 Visualization of Available Doctors")
        generate_plots(available_doctors, data)
    
    if show_model_details and has_two_classes and not available_doctors.empty:
        st.header("🧠 Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cross-Validation Scores")
            if eval_metrics['cv_scores'] is not None:
                cv_scores = eval_metrics['cv_scores']
                st.metric("Mean AUC-ROC", f"{eval_metrics['mean_cv_score']:.4f}")
                cv_df = pd.DataFrame({'Fold': range(1, len(cv_scores)+1), 'AUC-ROC': cv_scores})
                fig_cv = px.bar(cv_df, x='Fold', y='AUC-ROC', title="Cross-Validation AUC-ROC Scores")
                st.plotly_chart(fig_cv, use_container_width=True)
            else:
                st.write("Cross-validation scores not available.")
        
        with col2:
            st.subheader("Precision-Recall Curve")
            if eval_metrics['precision'] is not None:
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=eval_metrics['recall'], 
                    y=eval_metrics['precision'],
                    mode='lines',
                    name=f'AP={eval_metrics["avg_precision"]:.3f}'
                ))
                fig_pr.update_layout(
                    title='Precision-Recall Curve',
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    yaxis=dict(range=[0, 1.05]),
                    xaxis=dict(range=[0, 1.05])
                )
                st.plotly_chart(fig_pr, use_container_width=True)
            else:
                st.write("Precision-Recall Curve not available.")
        
        if feature_importance is not None:
            st.subheader("Feature Importance")
            top_features = feature_importance.head(15)
            fig_imp = px.bar(
                top_features,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 15 Most Important Features",
                color='Importance',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)
    
    # Peak Availability Times Section
    st.header("⏰ Peak Availability Times")
    st.write("These are the top 5 hours when the most doctors are historically active, based on the dataset.")
    
    peak_times = get_peak_availability_times(data)
    
    # Display as a bar chart
    fig_peak = px.bar(
        peak_times,
        x='Time',
        y='Active Count',
        color='Active Count',
        title="Top 5 Peak Availability Hours",
        color_continuous_scale=px.colors.sequential.Teal,
        text=peak_times['Active Count'].astype(int)
    )
    fig_peak.update_traces(textposition='auto')
    fig_peak.update_layout(
        xaxis_title="Hour of Day (24-hour format)",
        yaxis_title="Number of Active Doctors",
        xaxis={'tickmode': 'array', 'tickvals': peak_times['Time'], 'ticktext': peak_times['Time']}
    )
    st.plotly_chart(fig_peak, use_container_width=True)
    
    # Display as a table
    st.subheader("Peak Times Table")
    st.dataframe(
        peak_times[['Time', 'Active Count']],
        column_config={
            "Time": st.column_config.TextColumn("Hour"),
            "Active Count": st.column_config.NumberColumn("Active Doctors", format="%d")
        },
        hide_index=True
    )
    
    # Dropdown for peak times download format selection
    peak_download_format = st.selectbox(
        "Select download format for peak times",
        options=["CSV", "Excel"],
        key="download_peak_format"
    )
    
    if peak_download_format == "CSV":
        csv_peak = peak_times.to_csv(index=False)
        st.download_button(
            label="Download Peak Times",
            data=csv_peak,
            file_name="peak_availability_times.csv",
            mime="text/csv"
        )
    else:  # Excel
        excel_buffer_peak = io.BytesIO()
        peak_times.to_excel(excel_buffer_peak, index=False)
        excel_data_peak = excel_buffer_peak.getvalue()
        st.download_button(
            label="Download Peak Times",
            data=excel_data_peak,
            file_name="peak_availability_times.xlsx",
            mime="application/vnd.ms-excel"
        )
    
    st.markdown("---")
    st.markdown("#### 🔍 Need more insights?")
    st.markdown("""
    - Try different times to compare doctor availability patterns
    - Adjust the likelihood threshold to find more potential respondents
    - Use the model details to understand what factors influence availability
    - Leverage peak times for planning survey campaigns
    """)

if __name__ == "__main__":
    main()