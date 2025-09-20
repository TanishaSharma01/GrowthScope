import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import json
from google import genai
from google.genai import types

# IMPORTANT: KEEP THIS COMMENT
# Follow these instructions when using this blueprint:
# - Note that the newest Gemini model series is "gemini-2.5-flash" or gemini-2.5-pro"
#   - do not change this unless explicitly requested by the user
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'fallback_secret_key')

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compute_aggregates(df):
    """Compute per-product aggregates with profit analysis"""
    product_agg = df.groupby('Product').agg({
        'Revenue': 'sum',
        'Cost': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    product_agg['Profit'] = product_agg['Revenue'] - product_agg['Cost']
    product_agg['Margin_Pct'] = (product_agg['Profit'] / product_agg['Revenue'] * 100).fillna(0)
    product_agg['Unit_Profit'] = (product_agg['Profit'] / product_agg['Quantity']).fillna(0)
    
    # Add brand and category info
    if 'Brand' in df.columns and 'Category' in df.columns:
        brand_category = df.groupby('Product')[['Brand', 'Category']].first()
        product_agg = product_agg.merge(brand_category, on='Product')
    
    return product_agg

def build_rankings(df):
    """Build top product rankings by category and brand"""
    rankings = {}
    
    if 'Category' in df.columns:
        # Top 3 products by category (by profit)
        category_products = df.groupby(['Category', 'Product']).agg({
            'Revenue': 'sum',
            'Cost': 'sum'
        }).reset_index()
        category_products['Profit'] = category_products['Revenue'] - category_products['Cost']
        
        rankings['by_category'] = {}
        for category in category_products['Category'].unique():
            cat_data = category_products[category_products['Category'] == category]
            top_products = cat_data.nlargest(3, 'Profit')[['Product', 'Profit']].to_dict('records')
            rankings['by_category'][category] = top_products
    
    if 'Brand' in df.columns:
        # Top 3 products by brand (by profit)
        brand_products = df.groupby(['Brand', 'Product']).agg({
            'Revenue': 'sum',
            'Cost': 'sum'
        }).reset_index()
        brand_products['Profit'] = brand_products['Revenue'] - brand_products['Cost']
        
        rankings['by_brand'] = {}
        for brand in brand_products['Brand'].unique():
            brand_data = brand_products[brand_products['Brand'] == brand]
            top_products = brand_data.nlargest(3, 'Profit')[['Product', 'Profit']].to_dict('records')
            rankings['by_brand'][brand] = top_products
    
    return rankings

def compute_profit_tiers(product_agg):
    """Categorize products into profit tiers"""
    tiers = {'High': [], 'Medium': [], 'Low': [], 'Negative': []}
    
    for _, row in product_agg.iterrows():
        margin = row['Margin_Pct']
        product_info = {
            'product': row['Product'],
            'margin': round(margin, 2),
            'profit': round(row['Profit'], 2)
        }
        
        if margin >= 30:
            tiers['High'].append(product_info)
        elif margin >= 15:
            tiers['Medium'].append(product_info)
        elif margin >= 0:
            tiers['Low'].append(product_info)
        else:
            tiers['Negative'].append(product_info)
    
    # Sort by profit within each tier
    for tier in tiers.values():
        tier.sort(key=lambda x: x['profit'], reverse=True)
    
    return tiers

def compute_reorders(df, defaults=None):
    """Compute reorder recommendations based on inventory data"""
    if defaults is None:
        defaults = {'LeadTimeDays': 7, 'SafetyStock': 15}
    
    recommendations = []
    
    # Check if inventory columns exist
    inventory_cols = ['StockOnHand', 'LeadTimeDays', 'SafetyStock']
    has_inventory = all(col in df.columns for col in inventory_cols)
    
    if not has_inventory:
        return recommendations, "Inventory data not available"
    
    # Calculate average daily sales per product
    product_sales = df.groupby('Product').agg({
        'Quantity': 'sum',
        'StockOnHand': 'first',
        'LeadTimeDays': 'first',
        'SafetyStock': 'first'
    }).reset_index()
    
    # Assume data spans multiple days for ADS calculation
    days_in_data = df['Date'].nunique()
    product_sales['ADS'] = product_sales['Quantity'] / max(days_in_data, 1)
    
    for _, row in product_sales.iterrows():
        ads = row['ADS']
        stock = row['StockOnHand']
        lead_time = row['LeadTimeDays']
        safety_stock = row['SafetyStock']
        
        # Reorder point calculation
        reorder_point = ads * lead_time + safety_stock
        
        if stock <= reorder_point:
            order_qty = max(0, ads * (lead_time + 7) - stock)  # 7 days additional buffer
            
            recommendations.append({
                'product': row['Product'],
                'current_stock': int(stock),
                'reorder_point': int(reorder_point),
                'recommended_order': int(order_qty),
                'ads': round(ads, 2),
                'reason': f"Stock ({int(stock)}) below reorder point ({int(reorder_point)})"
            })
    
    return recommendations, None

def analyze_sales_data(csv_file_path):
    """Process CSV file and calculate business insights"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        
        # Ensure required columns exist
        required_columns = ['Date', 'Product', 'Revenue', 'Cost', 'Quantity', 'Brand', 'Category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return None, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Optional inventory columns
        inventory_defaults = {'LeadTimeDays': 7, 'SafetyStock': 15}
        for col, default in inventory_defaults.items():
            if col not in df.columns:
                df[col] = default
        
        # Calculate basic insights
        total_sales = df['Revenue'].sum()
        total_cost = df['Cost'].sum()
        total_profit = total_sales - total_cost
        profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
        total_quantity = df['Quantity'].sum()
        avg_revenue_per_item = total_sales / total_quantity if total_quantity > 0 else 0
        
        # Find top-selling product by revenue
        product_revenue = df.groupby('Product')['Revenue'].sum()
        top_selling_product = product_revenue.idxmax()
        top_product_revenue = product_revenue.max()
        
        # Enhanced analytics using helper functions
        product_aggregates = compute_aggregates(df)
        rankings = build_rankings(df)
        profit_tiers = compute_profit_tiers(product_aggregates)
        reorder_recommendations, reorder_error = compute_reorders(df)
        
        # Brand and category summaries
        brand_summary = {}
        category_summary = {}
        
        if 'Brand' in df.columns:
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
        
        if 'Category' in df.columns:
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
            
            # Enhanced analytics
            'rankings': rankings,
            'profit_tiers': profit_tiers,
            'brand_summary': brand_summary,
            'category_summary': category_summary,
            'reorder_recommendations': reorder_recommendations,
            'reorder_error': reorder_error,
            'product_aggregates': product_aggregates.to_dict('records')
        }
        
        return insights, None
        
    except Exception as e:
        return None, f"Error processing CSV file: {str(e)}"

def generate_ai_strategies(insights, business_goal):
    """Generate AI-powered business strategies using Google Gemini"""
    try:
        # Create prompt for Gemini
        prompt = f"""
        Based on the following business analytics and goal, provide exactly 3 actionable business strategies using only existing resources:

        Business Analytics:
        - Total Sales: ${insights['total_sales']:,.2f}
        - Total Profit: ${insights['total_profit']:,.2f}
        - Profit Margin: {insights['profit_margin']}%
        - Top-Selling Product: {insights['top_selling_product']} (${insights['top_product_revenue']:,.2f})
        - Total Items Sold: {insights['total_quantity']}
        - Average Revenue per Item: ${insights['avg_revenue_per_item']:,.2f}

        Business Goal: {business_goal}

        Provide your response in JSON format with exactly this structure:
        {{
            "strategies": [
                {{"title": "Strategy 1 Title", "description": "Detailed actionable strategy description"}},
                {{"title": "Strategy 2 Title", "description": "Detailed actionable strategy description"}},
                {{"title": "Strategy 3 Title", "description": "Detailed actionable strategy description"}}
            ]
        }}

        Focus on practical, cost-effective strategies that can be implemented immediately using existing resources.
        """

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        content = response.text
        if content is None:
            return None, "No response content from Gemini"
        result = json.loads(content)
        return result['strategies'], None
        
    except Exception as e:
        return None, f"Error generating AI strategies: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'csv_file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['csv_file']
        business_goal = request.form.get('business_goal', '').strip()
        
        # Validate inputs
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if not business_goal:
            flash('Please enter a business goal')
            return redirect(request.url)
        
        if file and file.filename and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            try:
                # Analyze sales data
                insights, error = analyze_sales_data(filepath)
                if error:
                    flash(f'Error analyzing data: {error}')
                    return redirect(request.url)
                
                # Generate AI strategies
                strategies, ai_error = generate_ai_strategies(insights, business_goal)
                if ai_error:
                    flash(f'Error generating strategies: {ai_error}')
                    # Still show insights even if AI fails
                    strategies = []
                
                return render_template('index.html', 
                                     insights=insights, 
                                     strategies=strategies, 
                                     business_goal=business_goal)
            finally:
                # Clean up uploaded file - always execute this
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass  # Ignore if file is already deleted
        else:
            flash('Invalid file type. Please upload a CSV file.')
            return redirect(request.url)
    
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)