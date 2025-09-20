import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import json
from datetime import datetime, timedelta
from google import genai
from google.genai import types
import tempfile

# Initialize Gemini client with error handling
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables!")
    gemini_client = None
else:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini API client initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing Gemini client: {e}")
        gemini_client = None

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'fallback_secret_key_for_development')

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def filter_data_by_date_range(df, date_filter, start_date=None, end_date=None):
    """Filter dataframe based on date range selection"""
    try:
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Remove rows with invalid dates
        df = df.dropna(subset=['Date'])

        if date_filter == 'all':
            return df

        # Get current date for relative filtering
        current_date = datetime.now()

        if date_filter == 'last_30_days':
            start_filter = current_date - timedelta(days=30)
            filtered_df = df[df['Date'] >= start_filter]
        elif date_filter == 'last_90_days':
            start_filter = current_date - timedelta(days=90)
            filtered_df = df[df['Date'] >= start_filter]
        elif date_filter == 'current_month':
            start_filter = current_date.replace(day=1)
            filtered_df = df[df['Date'] >= start_filter]
        elif date_filter == 'last_month':
            # Get first day of current month, then subtract to get last month
            first_day_current = current_date.replace(day=1)
            last_month_end = first_day_current - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            filtered_df = df[(df['Date'] >= last_month_start) & (df['Date'] <= last_month_end)]
        elif date_filter == 'current_quarter':
            # Get current quarter
            quarter = (current_date.month - 1) // 3 + 1
            quarter_start_month = (quarter - 1) * 3 + 1
            start_filter = current_date.replace(month=quarter_start_month, day=1)
            filtered_df = df[df['Date'] >= start_filter]
        elif date_filter == 'custom' and start_date and end_date:
            start_filter = pd.to_datetime(start_date)
            end_filter = pd.to_datetime(end_date)
            filtered_df = df[(df['Date'] >= start_filter) & (df['Date'] <= end_filter)]
        else:
            filtered_df = df

        return filtered_df

    except Exception as e:
        print(f"Error filtering data: {e}")
        return df


def get_date_range_summary(df, date_filter):
    """Get summary of the date range being analyzed"""
    if df.empty:
        return "No data available"

    min_date = df['Date'].min().strftime('%Y-%m-%d')
    max_date = df['Date'].max().strftime('%Y-%m-%d')
    total_days = (df['Date'].max() - df['Date'].min()).days + 1
    unique_dates = df['Date'].nunique()

    filter_names = {
        'all': 'All Time',
        'last_30_days': 'Last 30 Days',
        'last_90_days': 'Last 90 Days',
        'current_month': 'Current Month',
        'last_month': 'Last Month',
        'current_quarter': 'Current Quarter',
        'custom': 'Custom Range'
    }

    filter_name = filter_names.get(date_filter, 'Selected Period')

    return {
        'period_name': filter_name,
        'start_date': min_date,
        'end_date': max_date,
        'total_days': total_days,
        'data_points': len(df),
        'unique_dates': unique_dates
    }


def compute_monthly_trends(df):
    """Compute monthly revenue and profit trends"""
    try:
        # Ensure Date column is datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Create Year-Month column for grouping
        df['Year_Month'] = df['Date'].dt.to_period('M')

        # Group by month and aggregate
        monthly_data = df.groupby('Year_Month').agg({
            'Revenue': 'sum',
            'Cost': 'sum',
            'Quantity': 'sum'
        }).reset_index()

        # Calculate profit
        monthly_data['Profit'] = monthly_data['Revenue'] - monthly_data['Cost']

        # Convert Period to string for JSON serialization and sorting
        monthly_data['Month_str'] = monthly_data['Year_Month'].astype(str)
        monthly_data['Sort_Date'] = monthly_data['Year_Month'].dt.start_time

        # Sort by date
        monthly_data = monthly_data.sort_values('Sort_Date')

        # Format month labels for display (e.g., "Jan 2023")
        monthly_data['Month_Label'] = monthly_data['Year_Month'].dt.strftime('%b %Y')

        return monthly_data[['Month_str', 'Month_Label', 'Revenue', 'Cost', 'Profit', 'Quantity']].to_dict('records')

    except Exception as e:
        print(f"Error computing monthly trends: {str(e)}")
        return []


def analyze_sales_data(csv_file_path, date_filter='all', start_date=None, end_date=None):
    """Process CSV file and calculate business insights with date filtering"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path)

        # Handle both old and new data formats
        if 'Product_Name' in df.columns:
            # New transaction-based format
            required_columns = ['Date', 'Receipt_ID', 'Product_Name', 'Brand_Name', 'Category', 'Quantity',
                                'Selling_Price', 'Cost_Price']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return None, f"Missing required columns: {', '.join(missing_columns)}"

            # Map new column names to old format
            df['Product'] = df['Product_Name']
            df['Brand'] = df['Brand_Name']
            # Calculate total revenue and cost for each line item
            df['Revenue'] = df['Selling_Price'] * df['Quantity']
            df['Cost'] = df['Cost_Price'] * df['Quantity']

        else:
            # Old format
            required_columns = ['Date', 'Product', 'Revenue', 'Cost', 'Quantity']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return None, f"Missing required columns: {', '.join(missing_columns)}"

            # Optional columns with defaults for old format
            if 'Brand' not in df.columns:
                df['Brand'] = 'Unknown'
            if 'Category' not in df.columns:
                df['Category'] = 'Unknown'

        # Apply date filtering
        df = filter_data_by_date_range(df, date_filter, start_date, end_date)

        if df.empty:
            return None, "No data available for the selected date range"

        # Get date range summary
        date_summary = get_date_range_summary(df, date_filter)

        # Calculate basic insights
        total_sales = df['Revenue'].sum()
        total_cost = df['Cost'].sum()
        total_profit = total_sales - total_cost
        profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
        total_quantity = df['Quantity'].sum()
        avg_revenue_per_item = total_sales / total_quantity if total_quantity > 0 else 0

        # Find top-selling product by revenue
        if not df.empty:
            product_revenue = df.groupby('Product')['Revenue'].sum()
            top_selling_product = product_revenue.idxmax()
            top_product_revenue = product_revenue.max()
        else:
            top_selling_product = "N/A"
            top_product_revenue = 0

        # Monthly trends (replacing daily trends)
        monthly_trends = compute_monthly_trends(df)

        # Product performance
        product_agg = df.groupby('Product').agg({
            'Revenue': 'sum',
            'Cost': 'sum',
            'Quantity': 'sum'
        }).reset_index()
        product_agg['Profit'] = product_agg['Revenue'] - product_agg['Cost']
        product_agg['Margin_Pct'] = (product_agg['Profit'] / product_agg['Revenue'] * 100).fillna(0)

        # Transaction-level analytics (if Receipt_ID exists)
        transaction_metrics = {}
        if 'Receipt_ID' in df.columns:
            # Calculate per-receipt metrics
            receipt_agg = df.groupby('Receipt_ID').agg({
                'Revenue': 'sum',
                'Cost': 'sum',
                'Quantity': 'sum'
            }).reset_index()
            receipt_agg['Profit'] = receipt_agg['Revenue'] - receipt_agg['Cost']
            receipt_agg['Items_Per_Receipt'] = df.groupby('Receipt_ID').size().values

            transaction_metrics = {
                'total_receipts': len(receipt_agg),
                'avg_receipt_value': receipt_agg['Revenue'].mean(),
                'avg_items_per_receipt': receipt_agg['Items_Per_Receipt'].mean(),
                'avg_receipt_profit': receipt_agg['Profit'].mean(),
                'largest_receipt': receipt_agg['Revenue'].max(),
                'receipt_profit_margin': (receipt_agg['Profit'].sum() / receipt_agg['Revenue'].sum() * 100) if
                receipt_agg['Revenue'].sum() > 0 else 0
            }

        # Brand and category summaries
        brand_summary = {}
        category_summary = {}

        if 'Brand' in df.columns and not df.empty:
            brand_data = df.groupby('Brand').agg({
                'Revenue': 'sum',
                'Cost': 'sum',
                'Quantity': 'sum'
            }).reset_index()
            brand_data['Profit'] = brand_data['Revenue'] - brand_data['Cost']
            brand_data['Margin'] = ((brand_data['Profit'] / brand_data['Revenue']) * 100).fillna(0)
            brand_summary = {row['Brand']: {
                'Revenue': round(row['Revenue'], 2),
                'Cost': round(row['Cost'], 2),
                'Profit': round(row['Profit'], 2),
                'Margin': round(row['Margin'], 2),
                'Quantity': int(row['Quantity'])
            } for _, row in brand_data.iterrows()}

        if 'Category' in df.columns and not df.empty:
            category_data = df.groupby('Category').agg({
                'Revenue': 'sum',
                'Cost': 'sum',
                'Quantity': 'sum'
            }).reset_index()
            category_data['Profit'] = category_data['Revenue'] - category_data['Cost']
            category_data['Margin'] = ((category_data['Profit'] / category_data['Revenue']) * 100).fillna(0)
            category_summary = {row['Category']: {
                'Revenue': round(row['Revenue'], 2),
                'Cost': round(row['Cost'], 2),
                'Profit': round(row['Profit'], 2),
                'Margin': round(row['Margin'], 2),
                'Quantity': int(row['Quantity'])
            } for _, row in category_data.iterrows()}

        insights = {
            # Basic metrics
            'total_sales': total_sales,
            'total_cost': total_cost,
            'total_profit': total_profit,
            'profit_margin': round(profit_margin, 2),
            'top_selling_product': top_selling_product,
            'top_product_revenue': top_product_revenue,
            'total_quantity': total_quantity,
            'avg_revenue_per_item': round(avg_revenue_per_item, 2),

            # Date filtering info
            'date_summary': date_summary,

            # Time series data - NOW MONTHLY
            'monthly_trends': monthly_trends,

            # Transaction metrics
            'transaction_metrics': transaction_metrics,

            # Product analysis
            'product_aggregates': product_agg.to_dict('records') if not product_agg.empty else [],
            'brand_summary': brand_summary,
            'category_summary': category_summary,
        }

        return insights, None

    except Exception as e:
        return None, f"Error processing CSV file: {str(e)}"


def save_temp_file(file):
    """Save uploaded file to temporary location and return path"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    file.save(temp_file.name)
    temp_file.close()
    return temp_file.name


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'csv_file' not in request.files:
            flash('No file selected')
            return redirect(request.url)

        file = request.files['csv_file']
        date_filter = request.form.get('date_filter', 'all')
        start_date = request.form.get('start_date', '')
        end_date = request.form.get('end_date', '')

        # Validate inputs
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        # Validate custom date range
        if date_filter == 'custom':
            if not start_date or not end_date:
                flash('Please select both start and end dates for custom range')
                return redirect(request.url)

        if file and file.filename and allowed_file(file.filename):
            # Save file to session for dashboard access
            try:
                temp_path = save_temp_file(file)
                session['csv_file_path'] = temp_path
                session['date_filter'] = date_filter
                session['start_date'] = start_date
                session['end_date'] = end_date

                # Redirect to Executive Dashboard
                return redirect(url_for('executive_dashboard'))

            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a CSV file.')
            return redirect(request.url)

    return render_template('upload.html')


@app.route('/dashboard/executive')
def executive_dashboard():
    if 'csv_file_path' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))

    try:
        insights, error = analyze_sales_data(
            session['csv_file_path'],
            session.get('date_filter', 'all'),
            session.get('start_date'),
            session.get('end_date')
        )

        if error:
            flash(f'Error analyzing data: {error}')
            return redirect(url_for('index'))

        return render_template('executive_dashboard.html', insights=insights)

    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}')
        return redirect(url_for('index'))


@app.route('/dashboard/financial')
def financial_dashboard():
    if 'csv_file_path' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))

    try:
        insights, error = analyze_sales_data(
            session['csv_file_path'],
            session.get('date_filter', 'all'),
            session.get('start_date'),
            session.get('end_date')
        )

        if error:
            flash(f'Error analyzing data: {error}')
            return redirect(url_for('index'))

        return render_template('financial_dashboard.html', insights=insights)

    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}')
        return redirect(url_for('index'))


@app.route('/dashboard/growth')
def growth_dashboard():
    if 'csv_file_path' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))

    try:
        insights, error = analyze_sales_data(
            session['csv_file_path'],
            session.get('date_filter', 'all'),
            session.get('start_date'),
            session.get('end_date')
        )

        if error:
            flash(f'Error analyzing data: {error}')
            return redirect(url_for('index'))

        return render_template('growth_dashboard.html', insights=insights)

    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}')
        return redirect(url_for('index'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)