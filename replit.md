# MSME Analytics Coach

## Overview

MSME Analytics Coach is a Flask-based web application designed to help Micro, Small, and Medium Enterprises (MSMEs) analyze their sales data and gain business insights. The application allows users to upload CSV files containing sales data and provides automated analysis including revenue calculations, profit margins, and top-selling product identification. The system integrates with OpenAI's API to provide AI-powered business recommendations and insights.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Technology**: HTML templates with embedded CSS styling
- **Design Pattern**: Server-side rendered templates using Flask's Jinja2 templating engine
- **Styling**: Custom CSS with gradient backgrounds and modern responsive design
- **User Interface**: Single-page upload interface with intuitive file upload functionality

### Backend Architecture
- **Framework**: Flask (Python web framework)
- **Architecture Pattern**: Monolithic application with route-based request handling
- **Data Processing**: Pandas for CSV file analysis and data manipulation
- **File Handling**: Secure file upload with validation for CSV files only
- **Session Management**: Flask's built-in session handling with configurable secret key

### Data Processing Pipeline
- **Input Validation**: Ensures uploaded files are CSV format with required columns (Date, Product, Revenue, Cost, Quantity)
- **Analytics Engine**: Calculates key business metrics including total sales, costs, profit margins, and product performance
- **Error Handling**: Comprehensive validation for missing columns and data integrity

### AI Integration
- **Provider**: OpenAI API integration using the latest GPT-5 model
- **Purpose**: Generates business insights and recommendations based on processed sales data
- **Implementation**: Direct API calls using the official OpenAI Python client

## External Dependencies

### Third-Party Services
- **OpenAI API**: GPT-5 model for generating business insights and recommendations
- **Authentication**: API key-based authentication for OpenAI services

### Python Libraries
- **Flask**: Core web framework for routing and request handling
- **Pandas**: Data analysis and CSV file processing
- **Werkzeug**: Secure filename handling and utilities
- **OpenAI**: Official Python client for OpenAI API integration

### Environment Configuration
- **OPENAI_API_KEY**: Required environment variable for OpenAI API access
- **SESSION_SECRET**: Optional environment variable for Flask session security (fallback provided)

### File System Dependencies
- **Upload Directory**: Local file system storage for temporary CSV file uploads
- **Supported Formats**: CSV files only with strict validation