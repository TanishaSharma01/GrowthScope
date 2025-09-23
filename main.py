import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
import json
import time
import pickle
from datetime import datetime, timedelta
from google import genai
from google.genai import types
import tempfile
import numpy as np
from datetime import datetime, timedelta

# Initialize Gemini client with error handling
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', "AIzaSyD27VX_Kjr8wmfw3icVwxtMWTLgGc5yw3Q")
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

# Force login for all routes except /login and static
from flask import request


@app.before_request
def require_login_first():
    # Allow login route and static files without session
    if request.endpoint in ('login', 'static'):
        return
    if request.path.startswith('/static/'):
        return
    if not session.get('logged_in'):
        return redirect(url_for('login', next=request.path))


# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def convert_numpy_types(obj):
    """Convert numpy data types to Python native types for JSON serialization - OPTIMIZED"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


def create_lightweight_insights(insights):
    """Create a lightweight version of insights for session storage"""
    if not insights:
        return None

    # Only store essential data for chat - remove large arrays and detailed product data
    lightweight = {
        'total_sales': float(insights.get('total_sales', 0)),
        'total_cost': float(insights.get('total_cost', 0)),
        'total_profit': float(insights.get('total_profit', 0)),
        'profit_margin': float(insights.get('profit_margin', 0)),
        'total_quantity': int(insights.get('total_quantity', 0)),
        'avg_revenue_per_item': float(insights.get('avg_revenue_per_item', 0)),
        'top_selling_product': str(insights.get('top_selling_product', '')),
        'top_product_revenue': float(insights.get('top_product_revenue', 0)),

        # Keep date summary
        'date_summary': insights.get('date_summary', {}),

        # Keep only top 5 products to reduce size
        'top_products': [],

        # Keep transaction metrics if available
        'transaction_metrics': insights.get('transaction_metrics', {}),

        # Keep brand/category summaries but limit to top 5
        'brand_summary': {},
        'category_summary': {},

        # Monthly trends - keep only last 12 months
        'monthly_trends': []
    }

    # Add top 5 products only
    if insights.get('product_aggregates'):
        sorted_products = sorted(
            insights['product_aggregates'],
            key=lambda x: x.get('Profit', 0),
            reverse=True
        )[:5]

        lightweight['top_products'] = [{
            'Product': p.get('Product', ''),
            'Revenue': float(p.get('Revenue', 0)),
            'Cost': float(p.get('Cost', 0)),
            'Profit': float(p.get('Profit', 0)),
            'Margin_Pct': float(p.get('Margin_Pct', 0)),
            'Quantity': int(p.get('Quantity', 0))
        } for p in sorted_products]

    # Add top 5 brands
    if insights.get('brand_summary'):
        brand_items = list(insights['brand_summary'].items())[:5]
        lightweight['brand_summary'] = {k: v for k, v in brand_items}

    # Add top 5 categories
    if insights.get('category_summary'):
        category_items = list(insights['category_summary'].items())[:5]
        lightweight['category_summary'] = {k: v for k, v in category_items}

    # Add last 12 months of trends only
    if insights.get('monthly_trends'):
        lightweight['monthly_trends'] = insights['monthly_trends'][-12:]

    return convert_numpy_types(lightweight)


def update_chat_insights_session(insights):
    """Helper function to update chat insights in session - OPTIMIZED"""
    if insights:
        # Use lightweight version to avoid session size issues
        lightweight_insights = create_lightweight_insights(insights)
        session['current_insights'] = lightweight_insights
        session['data_timestamp'] = datetime.now().isoformat()

        period_name = lightweight_insights.get('date_summary', {}).get('period_name', 'Unknown')
        product_count = len(lightweight_insights.get('top_products', []))

        print(f"✅ Updated chat insights in session - Period: {period_name}, Products: {product_count}")
    else:
        session.pop('current_insights', None)
        session.pop('data_timestamp', None)
        print("❌ Cleared chat insights from session")


def safe_cleanup_temp_files():
    """Safely cleanup temp files without affecting current operations"""
    try:
        temp_dir = tempfile.gettempdir()
        current_time = time.time()

        # Clean up growthscope temp files older than 1 hour
        for filename in os.listdir(temp_dir):
            if filename.startswith('growthscope_') and filename.endswith('.csv'):
                filepath = os.path.join(temp_dir, filename)
                file_age = current_time - os.path.getctime(filepath)

                # If file is older than 1 hour and not the current session file
                if (file_age > 3600 and
                        filepath != session.get('csv_file_path')):
                    try:
                        os.unlink(filepath)
                        print(f"🗑️ Cleaned up old temp file: {filepath}")
                    except Exception as e:
                        print(f"⚠️ Could not clean up {filepath}: {e}")
    except Exception as e:
        print(f"⚠️ Error in safe cleanup: {e}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def filter_data_by_date_range(df, date_filter, start_date=None, end_date=None):
    """Filter dataframe based on date range selection - FIXED VERSION"""
    try:
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Remove rows with invalid dates
        df = df.dropna(subset=['Date'])

        if df.empty:
            return df

        if date_filter == 'all':
            return df

        # Get the max date from the data instead of current date
        max_data_date = df['Date'].max()
        min_data_date = df['Date'].min()

        print(f"Data date range: {min_data_date.strftime('%Y-%m-%d')} to {max_data_date.strftime('%Y-%m-%d')}")

        if date_filter == 'last_30_days':
            start_filter = max_data_date - timedelta(days=30)
            filtered_df = df[df['Date'] >= start_filter]

        elif date_filter == 'last_90_days':
            start_filter = max_data_date - timedelta(days=90)
            filtered_df = df[df['Date'] >= start_filter]

        elif date_filter == 'current_month':
            # Current month relative to the latest data
            start_filter = max_data_date.replace(day=1)
            filtered_df = df[df['Date'] >= start_filter]

        elif date_filter == 'last_month':
            # Last month relative to the latest data
            current_month_start = max_data_date.replace(day=1)
            last_month_end = current_month_start - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            filtered_df = df[(df['Date'] >= last_month_start) & (df['Date'] <= last_month_end)]

        elif date_filter == 'current_quarter':
            # Current quarter relative to the latest data
            quarter = (max_data_date.month - 1) // 3 + 1
            quarter_start_month = (quarter - 1) * 3 + 1
            start_filter = max_data_date.replace(month=quarter_start_month, day=1)
            filtered_df = df[df['Date'] >= start_filter]

        elif date_filter == 'custom' and start_date and end_date:
            start_filter = pd.to_datetime(start_date)
            end_filter = pd.to_datetime(end_date)
            filtered_df = df[(df['Date'] >= start_filter) & (df['Date'] <= end_filter)]

        else:
            filtered_df = df

        print(f"Filter '{date_filter}' returned {len(filtered_df)} rows out of {len(df)} total rows")
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
        'last_30_days': 'Last 30 Days (from data)',
        'last_90_days': 'Last 90 Days (from data)',
        'current_month': 'Current Month (from data)',
        'last_month': 'Last Month (from data)',
        'current_quarter': 'Current Quarter (from data)',
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
    """Compute monthly revenue, profit trends, and median ticket cost"""
    try:
        if df.empty:
            return []

        # Ensure Date column is datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Create Year-Month column for grouping
        df['Year_Month'] = df['Date'].dt.to_period('M')

        # Group by month and aggregate basic metrics
        monthly_data = df.groupby('Year_Month').agg({
            'Revenue': 'sum',
            'Cost': 'sum',
            'Quantity': 'sum'
        }).reset_index()

        # Calculate profit
        monthly_data['Profit'] = monthly_data['Revenue'] - monthly_data['Cost']

        # Calculate median ticket cost if Receipt_ID exists
        if 'Receipt_ID' in df.columns:
            # Calculate total cost per receipt (ticket) for each month
            monthly_median_costs = []

            for year_month in monthly_data['Year_Month']:
                # Filter data for this month
                month_data = df[df['Year_Month'] == year_month]

                # Calculate total cost per receipt (ticket)
                if not month_data.empty:
                    receipt_costs = month_data.groupby('Receipt_ID')['Cost'].sum()
                    median_ticket_cost = receipt_costs.median()
                    monthly_median_costs.append(median_ticket_cost)
                else:
                    monthly_median_costs.append(0)

            monthly_data['Median_Ticket_Cost'] = monthly_median_costs
        else:
            # If no receipt data, use average cost per transaction based on daily averages
            monthly_median_costs = []

            for year_month in monthly_data['Year_Month']:
                month_data = df[df['Year_Month'] == year_month]

                if not month_data.empty:
                    # Group by date and calculate daily total costs, then find median
                    daily_costs = month_data.groupby('Date')['Cost'].sum()
                    median_daily_cost = daily_costs.median()
                    monthly_median_costs.append(median_daily_cost)
                else:
                    monthly_median_costs.append(0)

            monthly_data['Median_Ticket_Cost'] = monthly_median_costs

        # Convert Period to string for JSON serialization and sorting
        monthly_data['Month_str'] = monthly_data['Year_Month'].astype(str)
        monthly_data['Sort_Date'] = monthly_data['Year_Month'].dt.start_time

        # Sort by date
        monthly_data = monthly_data.sort_values('Sort_Date')

        # Format month labels for display (e.g., "Jan 2023")
        monthly_data['Month_Label'] = monthly_data['Year_Month'].dt.strftime('%b %Y')

        result = monthly_data[
            ['Month_str', 'Month_Label', 'Revenue', 'Cost', 'Profit', 'Quantity', 'Median_Ticket_Cost']].to_dict(
            'records')

        # Convert numpy types to Python types
        return convert_numpy_types(result)

    except Exception as e:
        print(f"Error computing monthly trends: {str(e)}")
        return []


def analyze_sales_data(csv_file_path, date_filter='all', start_date=None, end_date=None):
    """Process CSV file and calculate business insights with date filtering"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} rows from CSV")

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
            return None, f"No data available for the selected date range '{date_filter}'. Try selecting 'All Time' or a different date range."

        print(f"After filtering: {len(df)} rows remaining")

        # Get date range summary
        date_summary = get_date_range_summary(df, date_filter)

        # Calculate basic insights
        total_sales = float(df['Revenue'].sum())
        total_cost = float(df['Cost'].sum())
        total_profit = total_sales - total_cost
        profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
        total_quantity = int(df['Quantity'].sum())
        avg_revenue_per_item = total_sales / total_quantity if total_quantity > 0 else 0

        # Find top-selling product by revenue
        if not df.empty:
            product_revenue = df.groupby('Product')['Revenue'].sum()
            top_selling_product = product_revenue.idxmax()
            top_product_revenue = float(product_revenue.max())
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
                'total_receipts': int(len(receipt_agg)),
                'avg_receipt_value': float(receipt_agg['Revenue'].mean()),
                'avg_items_per_receipt': float(receipt_agg['Items_Per_Receipt'].mean()),
                'avg_receipt_profit': float(receipt_agg['Profit'].mean()),
                'largest_receipt': float(receipt_agg['Revenue'].max()),
                'receipt_profit_margin': float((receipt_agg['Profit'].sum() / receipt_agg['Revenue'].sum() * 100) if
                                               receipt_agg['Revenue'].sum() > 0 else 0)
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
                'Revenue': float(row['Revenue']),
                'Cost': float(row['Cost']),
                'Profit': float(row['Profit']),
                'Margin': float(row['Margin']),
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
                'Revenue': float(row['Revenue']),
                'Cost': float(row['Cost']),
                'Profit': float(row['Profit']),
                'Margin': float(row['Margin']),
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

            # Product analysis - Convert to JSON serializable format
            'product_aggregates': convert_numpy_types(product_agg.to_dict('records')) if not product_agg.empty else [],
            'brand_summary': brand_summary,
            'category_summary': category_summary,

            # Store raw dataframe for chat functionality
            'raw_data': df
        }

        return insights, None

    except Exception as e:
        print(f"Error in analyze_sales_data: {e}")
        return None, f"Error processing CSV file: {str(e)}"


def generate_data_summary_for_ai(insights):
    """Generate a comprehensive data summary for AI analysis - WORKS WITH LIGHTWEIGHT DATA"""
    if not insights:
        return "No data available for analysis."

    summary = f"""
Business Data Summary:
- Total Revenue: ${insights['total_sales']:,.2f}
- Total Cost: ${insights['total_cost']:,.2f}
- Total Profit: ${insights['total_profit']:,.2f}
- Profit Margin: {insights['profit_margin']:.1f}%
- Total Quantity Sold: {insights['total_quantity']}
- Average Revenue per Item: ${insights['avg_revenue_per_item']:.2f}
- Top Selling Product: {insights['top_selling_product']} (${insights['top_product_revenue']:,.2f})

Date Range: {insights['date_summary']['period_name']} 
({insights['date_summary']['start_date']} to {insights['date_summary']['end_date']})
- Total Days: {insights['date_summary']['total_days']}
- Data Points: {insights['date_summary']['data_points']}
"""

    # Add transaction metrics if available
    if insights.get('transaction_metrics'):
        tm = insights['transaction_metrics']
        summary += f"""
Transaction Analytics:
- Total Transactions: {tm.get('total_receipts', 'N/A')}
- Average Transaction Value: ${tm.get('avg_receipt_value', 0):.2f}
- Average Items per Transaction: {tm.get('avg_items_per_receipt', 0):.1f}
- Largest Transaction: ${tm.get('largest_receipt', 0):.2f}
"""

    # Add top products
    if insights.get('top_products'):
        summary += f"\nTop Products by Profit:\n"
        for i, product in enumerate(insights['top_products'], 1):
            summary += f"{i}. {product['Product']}: ${product['Profit']:,.0f} profit, {product['Margin_Pct']:.1f}% margin\n"

    # Add brand/category info if available
    if insights.get('brand_summary'):
        summary += f"\nTop Brands:\n"
        for brand, data in insights['brand_summary'].items():
            if brand != 'Unknown':
                summary += f"- {brand}: ${data['Revenue']:,.0f} revenue, {data['Margin']:.1f}% margin\n"

    return summary


def ask_ai_about_data(question, insights):
    """Use Gemini AI to answer questions about the business data"""
    if not gemini_client:
        return "AI service is not available. Please check your API configuration."

    try:
        # Generate comprehensive data summary
        data_summary = generate_data_summary_for_ai(insights)

        # Create the prompt for Gemini
        prompt = f"""
You are a business intelligence analyst helping a business owner understand their sales data. 
You have access to comprehensive business data and should provide actionable insights.

Business Data:
{data_summary}

User Question: {question}

Please provide a helpful, specific answer based on the data above. Focus on:
1. Direct answers to the question asked
2. Relevant insights from the data
3. Actionable recommendations when appropriate
4. Specific numbers and metrics to support your points

Keep your response conversational but professional, and aim for 2-4 paragraphs maximum.
"""

        # Call Gemini API
        response = gemini_client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1000,
            )
        )

        if response and response.text:
            return response.text.strip()
        else:
            return "I'm sorry, I couldn't generate a response. Please try rephrasing your question."

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"I encountered an error while processing your question: {str(e)}"


def save_temp_file(file):
    """Save uploaded file to temporary location and return path - IMPROVED"""
    # Do safe cleanup of old files first (not aggressive cleanup)
    safe_cleanup_temp_files()

    # Create new temp file with timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f'_{timestamp}.csv',
        prefix='growthscope_'
    )
    file.save(temp_file.name)
    temp_file.close()
    print(f"📁 Saved new temp file: {temp_file.name}")
    return temp_file.name


def analyze_inventory_data(df):
    """Analyze inventory data and calculate stock requirements and restock dates"""
    try:
        if df.empty:
            return None

        # Check if we have stock data
        if 'Stock_For_Month' not in df.columns:
            return None

        # Get the latest month's data for current stock levels
        df['Date'] = pd.to_datetime(df['Date'])
        latest_month = df['Date'].max().to_period('M')
        current_month_data = df[df['Date'].dt.to_period('M') == latest_month]

        # Calculate monthly sales velocity for each product
        monthly_sales = df.groupby(['Product', df['Date'].dt.to_period('M')])['Quantity'].sum().reset_index()
        monthly_sales.columns = ['Product', 'Month', 'Monthly_Sales']

        # Calculate average monthly sales and stock requirements
        avg_monthly_sales = monthly_sales.groupby('Product')['Monthly_Sales'].mean().reset_index()
        avg_monthly_sales.columns = ['Product', 'Avg_Monthly_Sales']

        # Get current stock levels (latest month)
        current_stock = current_month_data.groupby('Product').agg({
            'Stock_For_Month': 'first',  # Assuming stock is same across all entries for a product in a month
            'Cost_Price': 'mean',
            'Selling_Price': 'mean'
        }).reset_index()

        # Merge sales velocity with current stock
        inventory_analysis = current_stock.merge(avg_monthly_sales, on='Product', how='left')
        inventory_analysis['Avg_Monthly_Sales'] = inventory_analysis['Avg_Monthly_Sales'].fillna(0)

        # Calculate required stock (2 months of average sales + safety buffer)
        inventory_analysis['Required_Stock'] = (inventory_analysis['Avg_Monthly_Sales'] * 2.5).round().astype(int)
        inventory_analysis['Current_Stock'] = inventory_analysis['Stock_For_Month']
        inventory_analysis['Shortage'] = np.maximum(0, inventory_analysis['Required_Stock'] - inventory_analysis[
            'Current_Stock'])

        # Calculate stock level categories
        def get_stock_level(row):
            if row['Current_Stock'] <= row['Avg_Monthly_Sales'] * 0.5:
                return 'Critical'
            elif row['Current_Stock'] <= row['Avg_Monthly_Sales']:
                return 'Low'
            elif row['Current_Stock'] <= row['Avg_Monthly_Sales'] * 2:
                return 'Adequate'
            else:
                return 'High'

        inventory_analysis['Stock_Level'] = inventory_analysis.apply(get_stock_level, axis=1)

        # Calculate next restock date based on current stock and sales velocity
        def calculate_restock_date(row):
            if row['Avg_Monthly_Sales'] <= 0:
                return "No sales data"

            days_of_stock = (row['Current_Stock'] / (row['Avg_Monthly_Sales'] / 30))

            if days_of_stock <= 7:
                restock_days = 3  # Urgent
            elif days_of_stock <= 30:
                restock_days = int(days_of_stock * 0.7)  # Restock before running out
            else:
                restock_days = 30  # Monthly review

            restock_date = datetime.now() + timedelta(days=restock_days)
            return restock_date.strftime('%Y-%m-%d')

        def calculate_days_until_restock(row):
            if row['Avg_Monthly_Sales'] <= 0:
                return 999

            days_of_stock = (row['Current_Stock'] / (row['Avg_Monthly_Sales'] / 30))

            if days_of_stock <= 7:
                return 3
            elif days_of_stock <= 30:
                return int(days_of_stock * 0.7)
            else:
                return 30

        inventory_analysis['Next_Restock_Date'] = inventory_analysis.apply(calculate_restock_date, axis=1)
        inventory_analysis['Days_Until_Restock'] = inventory_analysis.apply(calculate_days_until_restock, axis=1)

        # Calculate stock value
        inventory_analysis['Stock_Value'] = inventory_analysis['Current_Stock'] * inventory_analysis['Cost_Price']

        # Prepare summary statistics
        total_products = len(inventory_analysis)
        low_stock_count = len(inventory_analysis[inventory_analysis['Stock_Level'].isin(['Low', 'Critical'])])
        critical_stock_count = len(inventory_analysis[inventory_analysis['Stock_Level'] == 'Critical'])
        total_stock_value = inventory_analysis['Stock_Value'].sum()

        # Stock distribution for charts
        stock_distribution = inventory_analysis['Stock_Level'].value_counts().to_dict()

        # Restock timeline
        restock_timeline = []
        for days in [3, 7, 14, 30]:
            count = len(inventory_analysis[inventory_analysis['Days_Until_Restock'] <= days])
            date = (datetime.now() + timedelta(days=days)).strftime('%m/%d')
            restock_timeline.append({'date': f"Next {days}d", 'count': count})

        # Convert to list of dictionaries for template
        products_list = []
        for _, row in inventory_analysis.iterrows():
            products_list.append({
                'name': row['Product'],
                'current_stock': int(row['Current_Stock']),
                'required_stock': int(row['Required_Stock']),
                'shortage': int(row['Shortage']),
                'stock_level': row['Stock_Level'],
                'next_restock_date': row['Next_Restock_Date'],
                'days_until_restock': int(row['Days_Until_Restock']),
                'stock_value': float(row['Stock_Value'])
            })

        return {
            'total_products': total_products,
            'low_stock_count': low_stock_count,
            'critical_stock_count': critical_stock_count,
            'total_stock_value': float(total_stock_value),
            'stock_distribution': stock_distribution,
            'restock_timeline': restock_timeline,
            'products': products_list
        }

    except Exception as e:
        print(f"Error analyzing inventory data: {e}")
        return None


# Routes
@app.route('/')
def root():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Accept any credentials for demo purposes
        if username and password:
            session['logged_in'] = True
            session['username'] = username
            next_url = request.args.get('next')
            return redirect(next_url or url_for('index'))
        else:
            flash('Please enter both username and password')

    return render_template('login.html')


@app.route('/home', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login', next=request.path))

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
            try:
                temp_path = save_temp_file(file)

                # Clear old session data before setting new
                session.pop('current_insights', None)

                # Set new session data
                session['csv_file_path'] = temp_path
                session['date_filter'] = date_filter
                session['start_date'] = start_date
                session['end_date'] = end_date
                session['demo_data'] = False

                # Immediately analyze data and update chat session
                insights, error = analyze_sales_data(temp_path, date_filter, start_date, end_date)
                if insights and not error:
                    update_chat_insights_session(insights)
                    print(
                        f"✅ Successfully loaded uploaded data with {len(insights.get('product_aggregates', []))} products")
                else:
                    session.pop('current_insights', None)
                    print(f"❌ Failed to analyze uploaded data: {error}")

                return redirect(url_for('executive_dashboard'))

            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a CSV file.')
            return redirect(request.url)

    return render_template('upload.html')


@app.route('/load-demo-data', methods=['POST'])
def load_demo_data():
    """Load demo CSV data"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    try:
        demo_type = request.form.get('demo_type')
        date_filter = request.form.get('date_filter', 'all')
        start_date = request.form.get('start_date', '')
        end_date = request.form.get('end_date', '')

        # Map demo types to file names
        demo_files = {
            'sample_sales_enhanced': 'sample_sales_enhanced.csv',
            'synthetic_sales_100': 'synthetic_sales_100.csv',
            'supermarket_data': 'supermarket_data.csv'
        }

        if demo_type not in demo_files:
            flash('Invalid demo data type selected')
            return redirect(url_for('index'))

        demo_file_path = demo_files[demo_type]

        if not os.path.exists(demo_file_path):
            flash(f'Demo file {demo_file_path} not found.')
            return redirect(url_for('index'))

        # Only do safe cleanup, don't be aggressive
        safe_cleanup_temp_files()

        # Clear old session data before setting new
        session.pop('current_insights', None)

        # Store demo file path in session
        session['csv_file_path'] = demo_file_path
        session['date_filter'] = date_filter
        session['start_date'] = start_date
        session['end_date'] = end_date
        session['demo_data'] = True

        # Immediately analyze demo data and update chat session
        insights, error = analyze_sales_data(demo_file_path, date_filter, start_date, end_date)
        if insights and not error:
            update_chat_insights_session(insights)
            print(
                f"✅ Successfully loaded demo data '{demo_type}' with {len(insights.get('product_aggregates', []))} products")
        else:
            session.pop('current_insights', None)
            print(f"❌ Failed to analyze demo data '{demo_type}': {error}")

        return redirect(url_for('executive_dashboard'))

    except Exception as e:
        print(f"Error in load_demo_data: {e}")
        flash(f'Error loading demo data: {str(e)}')
        return redirect(url_for('index'))


@app.route('/dashboard/executive')
def executive_dashboard():
    if 'csv_file_path' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))

    try:
        csv_path = session['csv_file_path']

        # Check if file exists before analyzing
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            flash('Data file not found. Please upload your data again.')
            session.pop('csv_file_path', None)
            session.pop('current_insights', None)
            return redirect(url_for('index'))

        insights, error = analyze_sales_data(
            csv_path,
            session.get('date_filter', 'all'),
            session.get('start_date'),
            session.get('end_date')
        )

        if error:
            flash(f'Error analyzing data: {error}')
            return redirect(url_for('index'))

        # Remove raw_data before passing to template
        template_insights = {k: v for k, v in insights.items() if k != 'raw_data'}

        return render_template('executive_dashboard.html', insights=template_insights)

    except Exception as e:
        print(f"Error in executive_dashboard: {e}")
        flash(f'Error loading dashboard: {str(e)}')
        return redirect(url_for('index'))


@app.route('/dashboard/financial')
def financial_dashboard():
    if 'csv_file_path' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))

    try:
        csv_path = session['csv_file_path']

        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            flash('Data file not found. Please upload your data again.')
            session.pop('csv_file_path', None)
            session.pop('current_insights', None)
            return redirect(url_for('index'))

        insights, error = analyze_sales_data(
            csv_path,
            session.get('date_filter', 'all'),
            session.get('start_date'),
            session.get('end_date')
        )

        if error:
            flash(f'Error analyzing data: {error}')
            return redirect(url_for('index'))

        # Remove raw_data before passing to template
        template_insights = {k: v for k, v in insights.items() if k != 'raw_data'}

        return render_template('financial_dashboard.html', insights=template_insights)

    except Exception as e:
        print(f"Error in financial_dashboard: {e}")
        flash(f'Error loading dashboard: {str(e)}')
        return redirect(url_for('index'))


@app.route('/dashboard/growth')
def growth_dashboard():
    if 'csv_file_path' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))

    try:
        csv_path = session['csv_file_path']

        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            flash('Data file not found. Please upload your data again.')
            session.pop('csv_file_path', None)
            session.pop('current_insights', None)
            return redirect(url_for('index'))

        insights, error = analyze_sales_data(
            csv_path,
            session.get('date_filter', 'all'),
            session.get('start_date'),
            session.get('end_date')
        )

        if error:
            flash(f'Error analyzing data: {error}')
            return redirect(url_for('index'))

        # Remove raw_data before passing to template
        template_insights = {k: v for k, v in insights.items() if k != 'raw_data'}

        return render_template('growth_dashboard.html', insights=template_insights)

    except Exception as e:
        print(f"Error in growth_dashboard: {e}")
        flash(f'Error loading dashboard: {str(e)}')
        return redirect(url_for('index'))


@app.route('/dashboard/chat')
def chat_dashboard():
    if 'csv_file_path' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))

    try:
        csv_path = session['csv_file_path']

        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"❌ File not found for chat: {csv_path}")
            flash('Data file not found. Please upload your data again.')
            session.pop('csv_file_path', None)
            session.pop('current_insights', None)
            return redirect(url_for('index'))

        # Try to use existing session insights first
        if 'current_insights' in session:
            template_insights = session['current_insights']
            print(f"📊 Using cached insights for chat dashboard")
        else:
            # Fallback to analyzing data fresh
            insights, error = analyze_sales_data(
                csv_path,
                session.get('date_filter', 'all'),
                session.get('start_date'),
                session.get('end_date')
            )

            if error:
                flash(f'Error analyzing data: {error}')
                return redirect(url_for('index'))

            # Update session with fresh insights
            update_chat_insights_session(insights)
            template_insights = session['current_insights']

        return render_template('chat_dashboard.html', insights=template_insights)

    except Exception as e:
        print(f"Error in chat_dashboard: {e}")
        flash(f'Error loading chat dashboard: {str(e)}')
        return redirect(url_for('index'))


@app.route('/dashboard/chat/ask', methods=['POST'])
def chat_ask():
    """Handle chat questions via AJAX"""
    if 'current_insights' not in session:
        print("❌ No current_insights in session")
        return jsonify({'success': False, 'error': 'No data available for analysis. Please refresh the page.'})

    try:
        data = request.get_json()
        question = data.get('question', '').strip()

        if not question:
            return jsonify({'success': False, 'error': 'Please provide a question'})

        if len(question) > 500:
            return jsonify({'success': False, 'error': 'Question is too long. Please keep it under 500 characters.'})

        # Get insights from session
        insights = session['current_insights']

        # Add debugging info
        print(
            f"📊 Chat analysis using data from: {insights.get('date_summary', {}).get('period_name', 'Unknown period')}")
        print(
            f"📊 Data range: {insights.get('date_summary', {}).get('start_date', 'Unknown')} to {insights.get('date_summary', {}).get('end_date', 'Unknown')}")
        print(f"📊 Total revenue in data: ${insights.get('total_sales', 0):,.2f}")
        print(f"❓ User question: {question}")

        # Generate AI response
        answer = ask_ai_about_data(question, insights)

        return jsonify({'success': True, 'answer': answer})

    except Exception as e:
        print(f"❌ Error in chat_ask: {e}")
        return jsonify({'success': False, 'error': 'An error occurred while processing your question'})


@app.route('/dashboard/trends')
def trends_dashboard():
    """Static trends dashboard with dummy regional market data"""
    try:
        # This is a static dashboard that doesn't require CSV data
        # But we'll check if user has uploaded data to show consistent navigation
        return render_template('trends_dashboard.html')

    except Exception as e:
        flash(f'Error loading trends dashboard: {str(e)}')
        return redirect(url_for('index'))


@app.route('/dashboard/inventory')
def inventory_dashboard():
    if 'csv_file_path' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))

    try:
        csv_path = session['csv_file_path']

        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            flash('Data file not found. Please upload your data again.')
            session.pop('csv_file_path', None)
            session.pop('current_insights', None)
            return redirect(url_for('index'))

        insights, error = analyze_sales_data(
            csv_path,
            session.get('date_filter', 'all'),
            session.get('start_date'),
            session.get('end_date')
        )

        if error:
            flash(f'Error analyzing data: {error}')
            return redirect(url_for('index'))

        # Analyze inventory data
        inventory_data = None
        if insights and 'raw_data' in insights:
            inventory_data = analyze_inventory_data(insights['raw_data'])

        # Remove raw_data before passing to template
        template_insights = {k: v for k, v in insights.items() if k != 'raw_data'} if insights else None

        return render_template('inventory_dashboard.html',
                               insights=template_insights,
                               inventory_data=inventory_data)

    except Exception as e:
        print(f"Error in inventory_dashboard: {e}")
        flash(f'Error loading inventory dashboard: {str(e)}')
        return redirect(url_for('index'))


# Debug routes
@app.route('/debug/session')
def debug_session():
    """Debug route to check session data"""
    current_insights = session.get('current_insights', {})
    csv_path = session.get('csv_file_path', '')

    # Calculate session size
    session_size = 0
    try:
        session_size = len(pickle.dumps(dict(session)))
    except:
        session_size = 0

    return jsonify({
        'logged_in': session.get('logged_in'),
        'has_csv_path': 'csv_file_path' in session,
        'csv_file_path': csv_path,
        'file_exists': os.path.exists(csv_path) if csv_path else False,
        'file_size_mb': round(os.path.getsize(csv_path) / 1024 / 1024, 2) if csv_path and os.path.exists(
            csv_path) else 0,
        'has_insights': 'current_insights' in session,
        'insights_period': current_insights.get('date_summary', {}).get('period_name', 'Unknown'),
        'insights_date_range': f"{current_insights.get('date_summary', {}).get('start_date', 'Unknown')} to {current_insights.get('date_summary', {}).get('end_date', 'Unknown')}",
        'insights_revenue': current_insights.get('total_sales', 0),
        'insights_products': len(current_insights.get('top_products', [])),
        'data_timestamp': session.get('data_timestamp', 'Unknown'),
        'session_size_bytes': session_size,
        'session_size_kb': round(session_size / 1024, 2),
        'session_over_limit': session_size > 4093,
        'gemini_client_status': gemini_client is not None,
        'date_filter': session.get('date_filter'),
        'start_date': session.get('start_date'),
        'end_date': session.get('end_date'),
        'demo_data': session.get('demo_data', False),
        'session_keys': list(session.keys())
    })


@app.route('/debug/cleanup-session')
def cleanup_session():
    """Debug route to cleanup session"""
    try:
        # Safe cleanup of temp files
        safe_cleanup_temp_files()

        # Clear session data
        session.pop('current_insights', None)
        session.pop('csv_file_path', None)
        session.pop('data_timestamp', None)

        return jsonify({
            'success': True,
            'message': 'Session cleaned up successfully',
            'remaining_keys': list(session.keys())
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/refresh-chat-insights', methods=['POST'])
def refresh_chat_insights():
    """API endpoint to refresh chat insights"""
    if 'csv_file_path' not in session:
        return jsonify({'success': False, 'error': 'No data file in session'})

    try:
        csv_path = session['csv_file_path']

        if not os.path.exists(csv_path):
            return jsonify({'success': False, 'error': 'Data file not found'})

        insights, error = analyze_sales_data(
            csv_path,
            session.get('date_filter', 'all'),
            session.get('start_date'),
            session.get('end_date')
        )

        if error:
            return jsonify({'success': False, 'error': error})

        update_chat_insights_session(insights)

        return jsonify({
            'success': True,
            'message': 'Chat insights refreshed successfully',
            'period': insights.get('date_summary', {}).get('period_name', 'Unknown'),
            'revenue': insights.get('total_sales', 0),
            'products': len(insights.get('product_aggregates', []))
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# Create login template
@app.route('/create-login-template')
def create_login_template():
    """Create a basic login template if it doesn't exist"""
    login_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - GrowthScope</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f5f6fa; margin: 0; padding: 50px; }
        .container { max-width: 400px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; }
        input[type="text"], input[type="password"] { width: 100%; padding: 12px; border: 2px solid #e9ecef; border-radius: 8px; font-size: 16px; }
        button { width: 100%; background: #667eea; color: white; padding: 15px; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; }
        button:hover { background: #5a6fd8; }
        .flash-messages { margin-bottom: 20px; }
        .flash-message { padding: 15px; background: #f8d7da; color: #721c24; border-radius: 8px; margin-bottom: 10px; }
        h1 { text-align: center; color: #2c3e50; margin-bottom: 30px; }
        .demo-note { background: #e8f5e8; padding: 15px; border-radius: 8px; margin-bottom: 20px; color: #2d5a2d; text-align: center; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>GrowthScope Login</h1>
        <div class="demo-note">Demo Mode: Enter any username and password to login</div>

        {% with messages = get_flashed_messages() %}
            {% if messages %}
                <div class="flash-messages">
                    {% for message in messages %}
                        <div class="flash-message">{{ message }}</div>
                    {% endfor %}
                </div>
            {% endif %}
        {% endwith %}

        <form method="POST">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" name="username" required>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit">Login</button>
        </form>
    </div>
</body>
</html>"""

    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    # Write login template
    with open('templates/login.html', 'w') as f:
        f.write(login_html)

    return "Login template created successfully!"


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)