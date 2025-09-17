"""
Data quality validation and monitoring framework.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

@dataclass
class DataQualityReport:
    """Data quality report structure."""
    timestamp: datetime
    dataset_name: str
    total_rows: int
    total_columns: int
    missing_data: Dict[str, float]
    data_types: Dict[str, str]
    duplicates: int
    outliers: Dict[str, int]
    quality_score: float
    issues: List[str]
    recommendations: List[str]

class DataQualityValidator:
    """Comprehensive data quality validation framework."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.issues = []
        self.recommendations = []
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str = "dataset") -> DataQualityReport:
        """Perform comprehensive data quality validation."""
        self.logger.info(f"Starting data quality validation for {dataset_name}")
        
        #Reset issues and recommendations
        self.issues = []
        self.recommendations = []
        
        #Basic dataset info
        total_rows, total_columns = df.shape
        
        #Missing data analysis
        missing_data = self._analyze_missing_data(df)
        
        #Data types analysis
        data_types = self._analyze_data_types(df)
        
        #Duplicate analysis
        duplicates = self._analyze_duplicates(df)
        
        #Outlier analysis
        outliers = self._analyze_outliers(df)
        
        #Business rule validation
        self._validate_business_rules(df)
        
        #Calculate quality score
        quality_score = self._calculate_quality_score(df, missing_data, duplicates, outliers)
        
        return DataQualityReport(
            timestamp=datetime.now(),
            dataset_name=dataset_name,
            total_rows=total_rows,
            total_columns=total_columns,
            missing_data=missing_data,
            data_types=data_types,
            duplicates=duplicates,
            outliers=outliers,
            quality_score=quality_score,
            issues=self.issues,
            recommendations=self.recommendations
        )
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze missing data patterns."""
        missing_data = {}
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            missing_data[column] = missing_percentage
            
            if missing_percentage > self.config.max_missing_percentage * 100:
                self.issues.append(f"High missing data in {column}: {missing_percentage:.2f}%")
                self.recommendations.append(f"Consider imputation or removal of {column}")
        
        return missing_data
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze data types."""
        data_types = {}
        for column in df.columns:
            data_types[column] = str(df[column].dtype)
        return data_types
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> int:
        """Analyze duplicate rows."""
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            self.issues.append(f"Found {duplicates} duplicate rows")
            self.recommendations.append("Consider removing duplicate rows")
        return duplicates
    
    def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze outliers in numerical columns."""
        outliers = {}
        
        for column in df.select_dtypes(include=[np.number]).columns:
            if column == self.config.target_column:
                continue
                
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.config.outlier_threshold * IQR
            upper_bound = Q3 + self.config.outlier_threshold * IQR
            
            outlier_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            outliers[column] = outlier_count
            
            if outlier_count > len(df) * 0.05:  # More than 5% outliers
                self.issues.append(f"High outlier count in {column}: {outlier_count}")
                self.recommendations.append(f"Investigate outliers in {column}")
        
        return outliers
    
    def _validate_business_rules(self, df: pd.DataFrame):
        """Validate business-specific rules."""
        #Price validation
        if self.config.target_column in df.columns:
            price_col = df[self.config.target_column]
            
            #Check for negative prices
            negative_prices = (price_col < 0).sum()
            if negative_prices > 0:
                self.issues.append(f"Found {negative_prices} negative prices")
                self.recommendations.append("Remove or correct negative prices")
            
            #Check for unrealistic prices
            low_prices = (price_col < self.config.min_price).sum()
            high_prices = (price_col > self.config.max_price).sum()
            
            if low_prices > 0:
                self.issues.append(f"Found {low_prices} prices below minimum threshold")
                self.recommendations.append("Review low price entries")
            
            if high_prices > 0:
                self.issues.append(f"Found {high_prices} prices above maximum threshold")
                self.recommendations.append("Review high price entries")
        
        #Date validation
        if 'Date_of_Journey' in df.columns:
            try:
                dates = pd.to_datetime(df['Date_of_Journey'], dayfirst=True, errors='coerce')
                invalid_dates = dates.isnull().sum()
                if invalid_dates > 0:
                    self.issues.append(f"Found {invalid_dates} invalid dates")
                    self.recommendations.append("Fix invalid date entries")
            except:
                self.issues.append("Date column format issues detected")
                self.recommendations.append("Standardize date format")
        
        #Duration validation
        if 'Duration' in df.columns:
            #Check for unrealistic durations (negative or too long)
            duration_issues = 0
            for duration in df['Duration']:
                if isinstance(duration, str):
                    if 'h' in duration and 'm' in duration:
                        try:
                            hours = int(duration.split('h')[0].strip())
                            minutes = int(duration.split('h')[1].split('m')[0].strip())
                            total_minutes = hours * 60 + minutes
                            if total_minutes <= 0 or total_minutes > 1440:  # More than 24 hours
                                duration_issues += 1
                        except:
                            duration_issues += 1
            
            if duration_issues > 0:
                self.issues.append(f"Found {duration_issues} invalid duration entries")
                self.recommendations.append("Fix duration format issues")
    
    def _calculate_quality_score(self, df: pd.DataFrame, missing_data: Dict, 
                                duplicates: int, outliers: Dict) -> float:
        """Calculate overall data quality score (0-100)."""
        score = 100.0
        
        #Deduct for missing data
        avg_missing = np.mean(list(missing_data.values()))
        score -= min(avg_missing * 2, 30)  # Max 30 points deduction
        
        #Deduct for duplicates
        duplicate_percentage = (duplicates / len(df)) * 100
        score -= min(duplicate_percentage * 5, 20)  # Max 20 points deduction
        
        #Deduct for outliers
        total_outliers = sum(outliers.values())
        if len(outliers) > 0:
            outlier_percentage = (total_outliers / (len(df) * len(outliers))) * 100
            score -= min(outlier_percentage, 20)  # Max 20 points deduction
        
        #Deduct for business rule violations
        score -= len(self.issues) * 2  # 2 points per issue
        
        return max(score, 0)
    
    def generate_quality_report(self, report: DataQualityReport) -> str:
        """Generate a formatted quality report."""
        report_text = f"""
=== DATA QUALITY REPORT ===
Dataset: {report.dataset_name}
Timestamp: {report.timestamp}
Total Rows: {report.total_rows:,}
Total Columns: {report.total_columns}
Quality Score: {report.quality_score:.1f}/100

=== MISSING DATA ANALYSIS ===
"""
        for column, percentage in report.missing_data.items():
            status = "⚠️" if percentage > self.config.max_missing_percentage * 100 else "✅"
            report_text += f"{status} {column}: {percentage:.2f}%\n"
        
        report_text += f"""
=== DATA TYPES ===
"""
        for column, dtype in report.data_types.items():
            report_text += f"• {column}: {dtype}\n"
        
        report_text += f"""
=== DUPLICATES ===
Total Duplicate Rows: {report.duplicates}

=== OUTLIERS ===
"""
        for column, count in report.outliers.items():
            if count > 0:
                report_text += f"• {column}: {count} outliers\n"
        
        if report.issues:
            report_text += "\n=== ISSUES FOUND ===\n"
            for issue in report.issues:
                report_text += f"⚠️ {issue}\n"
        
        if report.recommendations:
            report_text += "\n=== RECOMMENDATIONS ===\n"
            for rec in report.recommendations:
                report_text += f"💡 {rec}\n"
        
        return report_text

class DataProfiler:
    """Advanced data profiling and statistical analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile."""
        profile = {
            'basic_info': self._get_basic_info(df),
            'numerical_summary': self._get_numerical_summary(df),
            'categorical_summary': self._get_categorical_summary(df),
            'correlation_analysis': self._get_correlation_analysis(df),
            'distribution_analysis': self._get_distribution_analysis(df)
        }
        return profile
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict:
        """Get basic dataset information."""
        return {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict()
        }
    
    def _get_numerical_summary(self, df: pd.DataFrame) -> Dict:
        """Get numerical columns summary."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) == 0:
            return {}
        
        summary = df[numerical_cols].describe()
        return summary.to_dict()
    
    def _get_categorical_summary(self, df: pd.DataFrame) -> Dict:
        """Get categorical columns summary."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        summary = {}
        
        for col in categorical_cols:
            summary[col] = {
                'unique_count': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'frequency': df[col].value_counts().head().to_dict()
            }
        
        return summary
    
    def _get_correlation_analysis(self, df: pd.DataFrame) -> Dict:
        """Get correlation analysis."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            return {}
        
        correlation_matrix = df[numerical_cols].corr()
        return correlation_matrix.to_dict()
    
    def _get_distribution_analysis(self, df: pd.DataFrame) -> Dict:
        """Get distribution analysis for numerical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        distributions = {}
        
        for col in numerical_cols:
            distributions[col] = {
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'normality_test': self._test_normality(df[col])
            }
        
        return distributions
    
    def _test_normality(self, series: pd.Series) -> Dict:
        """Test for normality using Shapiro-Wilk test."""
        from scipy import stats
        
        try:
            #Sample for large datasets
            if len(series) > 5000:
                sample = series.sample(5000, random_state=42)
            else:
                sample = series.dropna()
            
            if len(sample) < 3:
                return {'statistic': None, 'p_value': None, 'is_normal': False}
            
            statistic, p_value = stats.shapiro(sample)
            return {
                'statistic': statistic,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        except:
            return {'statistic': None, 'p_value': None, 'is_normal': False}
