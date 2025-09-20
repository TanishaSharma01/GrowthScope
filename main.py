import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import json
from openai import OpenAI

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'fallback_secret_key')

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_sales_data(csv_file_path):
    """Process CSV file and calculate business insights"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        
        # Ensure required columns exist
        required_columns = ['Date', 'Product', 'Revenue', 'Cost', 'Quantity']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return None, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Calculate insights
        total_sales = df['Revenue'].sum()
        total_cost = df['Cost'].sum()
        total_profit = total_sales - total_cost
        profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
        
        # Find top-selling product by revenue
        product_revenue = df.groupby('Product')['Revenue'].sum()
        top_selling_product = product_revenue.idxmax()
        top_product_revenue = product_revenue.max()
        
        # Additional insights
        total_quantity = df['Quantity'].sum()
        avg_revenue_per_item = total_sales / total_quantity if total_quantity > 0 else 0
        
        insights = {
            'total_sales': total_sales,
            'total_cost': total_cost,
            'total_profit': total_profit,
            'profit_margin': round(profit_margin, 2),
            'top_selling_product': top_selling_product,
            'top_product_revenue': top_product_revenue,
            'total_quantity': total_quantity,
            'avg_revenue_per_item': round(avg_revenue_per_item, 2)
        }
        
        return insights, None
        
    except Exception as e:
        return None, f"Error processing CSV file: {str(e)}"

def generate_ai_strategies(insights, business_goal):
    """Generate AI-powered business strategies using OpenAI"""
    try:
        # Create prompt for OpenAI
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

        response = openai_client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if content is None:
            return None, "No response content from OpenAI"
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
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return render_template('index.html', 
                                 insights=insights, 
                                 strategies=strategies, 
                                 business_goal=business_goal)
        else:
            flash('Invalid file type. Please upload a CSV file.')
            return redirect(request.url)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)