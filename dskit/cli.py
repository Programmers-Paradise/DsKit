"""
Command Line Interface for dskit
"""
import argparse
import sys
import pandas as pd
from dskit import dskit
from dskit.config import set_config, print_config

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='dskit - Data Science Toolkit Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  dskit eda data.csv --target churn                    # Quick EDA
  dskit profile data.csv --output report.html          # Generate profile report
  dskit health data.csv                                # Data health check
  dskit clean data.csv --output cleaned_data.csv       # Clean dataset
  dskit compare data.csv --target price --task regression  # Compare models
  dskit config --show                                  # Show current config
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # EDA command
    eda_parser = subparsers.add_parser('eda', help='Perform exploratory data analysis')
    eda_parser.add_argument('file', help='Input CSV file')
    eda_parser.add_argument('--target', help='Target column name')
    eda_parser.add_argument('--sample', type=int, help='Sample size for analysis')
    
    # Profile command
    profile_parser = subparsers.add_parser('profile', help='Generate automated profile report')
    profile_parser.add_argument('file', help='Input CSV file')
    profile_parser.add_argument('--output', default='profile_report.html', help='Output file name')
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Perform data health check')
    health_parser.add_argument('file', help='Input CSV file')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean dataset')
    clean_parser.add_argument('file', help='Input CSV file')
    clean_parser.add_argument('--output', required=True, help='Output CSV file')
    clean_parser.add_argument('--remove-outliers', action='store_true', help='Remove outliers')
    
    # Compare models command
    compare_parser = subparsers.add_parser('compare', help='Compare ML models')
    compare_parser.add_argument('file', help='Input CSV file')
    compare_parser.add_argument('--target', required=True, help='Target column name')
    compare_parser.add_argument('--task', choices=['classification', 'regression'], 
                               default='classification', help='ML task type')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    config_parser.add_argument('--reset', action='store_true', help='Reset to default configuration')
    config_parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'), 
                               help='Set configuration parameter')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show dataset information')
    info_parser.add_argument('file', help='Input CSV file')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        if args.command == 'eda':
            run_eda(args)
        elif args.command == 'profile':
            run_profile(args)
        elif args.command == 'health':
            run_health(args)
        elif args.command == 'clean':
            run_clean(args)
        elif args.command == 'compare':
            run_compare(args)
        elif args.command == 'config':
            run_config(args)
        elif args.command == 'info':
            run_info(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def run_eda(args):
    """Run EDA command."""
    print(f"Loading data from {args.file}...")
    kit = dskit.load(args.file)
    
    print("Performing comprehensive EDA...")
    kit.comprehensive_eda(target_col=args.target, sample_size=args.sample)

def run_profile(args):
    """Run profile command."""
    print(f"Loading data from {args.file}...")
    kit = dskit.load(args.file)
    
    print(f"Generating profile report: {args.output}...")
    kit.generate_profile_report(args.output)
    print(f"Profile report saved as {args.output}")

def run_health(args):
    """Run health check command."""
    print(f"Loading data from {args.file}...")
    kit = dskit.load(args.file)
    
    print("Performing data health check...")
    score = kit.data_health_check()
    print(f"\\nOverall Data Health Score: {score:.1f}/100")

def run_clean(args):
    """Run clean command."""
    print(f"Loading data from {args.file}...")
    kit = dskit.load(args.file)
    
    print("Cleaning dataset...")
    kit.clean()
    
    if args.remove_outliers:
        print("Removing outliers...")
        kit.remove_outliers()
    
    print(f"Saving cleaned data to {args.output}...")
    kit.save(args.output)
    print(f"Cleaned dataset saved as {args.output}")

def run_compare(args):
    """Run model comparison command."""
    print(f"Loading data from {args.file}...")
    kit = dskit.load(args.file)
    
    print("Preprocessing data...")
    kit.clean().auto_encode()
    
    print(f"Comparing models for {args.task} task...")
    results = kit.compare_models(args.target, task=args.task)
    
    print("\\nModel Comparison Results:")
    print(results)

def run_config(args):
    """Run config command."""
    if args.show:
        print_config()
    elif args.reset:
        from dskit.config import reset_config
        reset_config()
        print("Configuration reset to defaults.")
    elif args.set:
        key, value = args.set
        # Try to convert value to appropriate type
        try:
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif '.' in value:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string
        except:
            pass  # Keep as string
        
        set_config({key: value})
        print(f"Set {key} = {value}")

def run_info(args):
    """Run info command."""
    print(f"Loading data from {args.file}...")
    kit = dskit.load(args.file)
    
    print("\\nDataset Information:")
    print(f"Shape: {kit.df.shape}")
    print(f"Memory usage: {kit.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\\nData types:")
    print(kit.df.dtypes.value_counts())
    
    print("\\nBasic statistics:")
    stats = kit.basic_stats()
    print(stats.head(10))
    
    missing = kit.missing_summary()
    if not missing.empty:
        print("\\nMissing values:")
        print(missing.head(10))

if __name__ == '__main__':
    main()