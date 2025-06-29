"""
🎓 Yazd University Intelligent Assistant Frontend
=================================================

A professional Streamlit-based frontend for the Yazd University
Intelligent Assistant. Provides an intuitive and modern interface
for querying university professor information.

Features:
- Modern, responsive UI design
- Real-time query processing
- Query history and session management
- Advanced search options
- Professional styling and animations
- Comprehensive error handling
- Performance monitoring

Author: AI Assistant
Version: 1.0.0
Last Updated: 2024
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import base64
from io import BytesIO

# =============================================================================
# CONFIGURATION
# =============================================================================

# Application configuration
APP_CONFIG = {
    "name": "Yazd University Intelligent Assistant",
    "version": "1.0.0",
    "description": "دستیار هوشمند دانشگاه یزد",
    "backend_url": "http://localhost:8000",
    "api_endpoints": {
        "query": "/query",
        "health": "/health",
        "metrics": "/metrics",
        "info": "/info",
        "retrieval_methods": "/retrieval-methods"
    }
}

# Page configuration
st.set_page_config(
    page_title=APP_CONFIG["name"],
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def add_logo():
    """Add university logo to the sidebar."""
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #1f77b4; font-size: 24px; margin-bottom: 10px;">🎓</h1>
            <h2 style="color: #2c3e50; font-size: 18px; margin: 0;">دانشگاه یزد</h2>
            <p style="color: #7f8c8d; font-size: 14px; margin: 5px 0;">دستیار هوشمند</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def check_backend_health() -> bool:
    """Check if the backend is healthy and accessible."""
    try:
        response = requests.get(f"{APP_CONFIG['backend_url']}{APP_CONFIG['api_endpoints']['health']}", timeout=5)
        return response.status_code == 200
    except:
        return False

def format_processing_time(seconds: float) -> str:
    """Format processing time in a human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    else:
        return f"{seconds:.2f}s"

def create_confidence_gauge(confidence: float) -> go.Figure:
    """Create a confidence gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "اعتماد پاسخ", 'font': {'size': 16}},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_processing_time_chart(times: List[float]) -> go.Figure:
    """Create a processing time trend chart."""
    df = pd.DataFrame({
        'Query Number': range(1, len(times) + 1),
        'Processing Time (s)': times
    })
    
    fig = px.line(df, x='Query Number', y='Processing Time (s)',
                  title="زمان پردازش درخواست‌ها",
                  labels={'Query Number': 'شماره درخواست', 'Processing Time (s)': 'زمان پردازش (ثانیه)'})
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title_font_size=12,
        yaxis_title_font_size=12
    )
    
    return fig

# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def initialize_session_state():
    """Initialize session state variables."""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = f"session_{int(time.time())}"
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = f"user_{int(time.time())}"
    
    if 'processing_times' not in st.session_state:
        st.session_state.processing_times = []
    
    if 'backend_status' not in st.session_state:
        st.session_state.backend_status = check_backend_health()

# =============================================================================
# API FUNCTIONS
# =============================================================================

def send_query(query: str, use_self_query: bool = True) -> Optional[Dict]:
    """Send query to the backend API."""
    try:
        payload = {
            "query": query,
            "user_id": st.session_state.user_id,
            "session_id": st.session_state.current_session_id,
            "use_self_query": use_self_query
        }
        
        response = requests.post(
            f"{APP_CONFIG['backend_url']}{APP_CONFIG['api_endpoints']['query']}",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"خطا در ارتباط با سرور: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("زمان انتظار برای پاسخ به پایان رسید. لطفاً دوباره تلاش کنید.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("خطا در اتصال به سرور. لطفاً اتصال اینترنت خود را بررسی کنید.")
        return None
    except Exception as e:
        st.error(f"خطای غیرمنتظره: {str(e)}")
        return None

def get_system_metrics() -> Optional[Dict]:
    """Get system metrics from the backend."""
    try:
        response = requests.get(
            f"{APP_CONFIG['backend_url']}{APP_CONFIG['api_endpoints']['metrics']}",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except:
        return None

def get_retrieval_methods() -> Optional[Dict]:
    """Get available retrieval methods information."""
    try:
        response = requests.get(
            f"{APP_CONFIG['backend_url']}{APP_CONFIG['api_endpoints']['retrieval_methods']}",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except:
        return None

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_header():
    """Render the main header."""
    st.markdown(
        """
        <div style="text-align: center; padding: 20px 0; background: linear-gradient(90deg, #1f77b4, #ff7f0e); border-radius: 10px; margin-bottom: 30px;">
            <h1 style="color: white; font-size: 2.5em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                🎓 دستیار هوشمند دانشگاه یزد
            </h1>
            <p style="color: white; font-size: 1.2em; margin: 10px 0 0 0; opacity: 0.9;">
                Yazd University Intelligent Assistant
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_query_form():
    """Render the main query form."""
    st.markdown("### 💬 پرسش خود را مطرح کنید")
    
    # Query input
    query = st.text_area(
        "سوال خود را اینجا بنویسید:",
        placeholder="مثال: ایمیل استاد جهانگرد چیست؟ یا برنامه کلاسی استاد محمدی را نشان بده",
        height=120,
        help="سوال خود را به زبان فارسی بنویسید. می‌توانید در مورد اساتید، برنامه کلاسی، سوابق تحصیلی و تخصص‌ها بپرسید."
    )
    
    # Advanced options
    with st.expander("⚙️ تنظیمات پیشرفته"):
        col1, col2 = st.columns(2)
        
        with col1:
            use_self_query = st.checkbox(
                "استفاده از جستجوی ساختاریافته",
                value=True,
                help="این گزینه از الگوریتم‌های پیشرفته برای تجزیه و تحلیل سوال استفاده می‌کند"
            )
        
        with col2:
            st.markdown("**نوع جستجو:**")
            if use_self_query:
                st.success("🔍 جستجوی ساختاریافته فعال")
            else:
                st.info("🔎 جستجوی استاندارد فعال")
    
    # Submit button
    submit_button = st.button(
        "🚀 ارسال پرسش",
        type="primary",
        use_container_width=True,
        disabled=not query.strip()
    )
    
    return query.strip(), use_self_query, submit_button

def render_response(response_data: Dict):
    """Render the query response."""
    st.markdown("### 📋 پاسخ")
    
    # Main answer
    with st.container():
        st.markdown(
            f"""
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;">
                <div style="font-size: 16px; line-height: 1.6; text-align: justify;">
                    {response_data['answer']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Response metadata
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "زمان پردازش",
            format_processing_time(response_data['processing_time'])
        )
    
    with col2:
        st.metric(
            "تعداد منابع",
            response_data['sources_count']
        )
    
    with col3:
        if response_data.get('confidence_score'):
            confidence = response_data['confidence_score']
            st.metric(
                "اعتماد پاسخ",
                f"{confidence * 100:.1f}%"
            )
    
    with col4:
        method_icon = "🔍" if response_data['retrieval_method'] == 'self-query' else "🔎"
        st.metric(
            "روش جستجو",
            f"{method_icon} {response_data['retrieval_method']}"
        )
    
    # Confidence gauge
    if response_data.get('confidence_score'):
        st.plotly_chart(create_confidence_gauge(response_data['confidence_score']), use_container_width=True)
    
    # Context sources
    if response_data['context']:
        with st.expander(f"📚 منابع استفاده شده ({len(response_data['context'])})"):
            for i, context in enumerate(response_data['context'], 1):
                st.markdown(f"**منبع {i}:**")
                st.text(context[:500] + "..." if len(context) > 500 else context)
                st.divider()

def render_query_history():
    """Render query history."""
    if st.session_state.query_history:
        st.markdown("### 📜 تاریخچه پرسش‌ها")
        
        for i, (query, response, timestamp) in enumerate(reversed(st.session_state.query_history[-10:]), 1):
            with st.expander(f"پرسش {len(st.session_state.query_history) - i + 1}: {query[:50]}..."):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**پرسش:** {query}")
                    st.markdown(f"**پاسخ:** {response['answer'][:200]}...")
                
                with col2:
                    st.markdown(f"**زمان:** {timestamp.strftime('%H:%M:%S')}")
                    st.markdown(f"**روش:** {response['retrieval_method']}")

def render_analytics():
    """Render analytics and metrics."""
    st.markdown("### 📊 آمار و تحلیل")
    
    # Get system metrics
    metrics = get_system_metrics()
    
    if metrics:
        app_metrics = metrics.get('application_metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "کل درخواست‌ها",
                app_metrics.get('total_queries', 0)
            )
        
        with col2:
            success_rate = app_metrics.get('success_rate', 0)
            st.metric(
                "نرخ موفقیت",
                f"{success_rate:.1f}%"
            )
        
        with col3:
            avg_time = app_metrics.get('average_processing_time', 0)
            st.metric(
                "میانگین زمان پردازش",
                format_processing_time(avg_time)
            )
        
        with col4:
            uptime = app_metrics.get('uptime', 0)
            st.metric(
                "زمان فعالیت",
                f"{uptime/3600:.1f}h"
            )
        
        # Processing time chart
        if st.session_state.processing_times:
            st.plotly_chart(create_processing_time_chart(st.session_state.processing_times), use_container_width=True)
    
    else:
        st.warning("اطلاعات آماری در دسترس نیست")

def render_sidebar():
    """Render the sidebar with additional features."""
    add_logo()
    
    st.sidebar.markdown("---")
    
    # Backend status
    st.sidebar.markdown("### 🔧 وضعیت سیستم")
    
    if st.session_state.backend_status:
        st.sidebar.success("✅ سرور فعال")
    else:
        st.sidebar.error("❌ سرور غیرفعال")
        if st.sidebar.button("🔄 بررسی مجدد"):
            st.session_state.backend_status = check_backend_health()
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Quick actions
    st.sidebar.markdown("### ⚡ اقدامات سریع")
    
    if st.sidebar.button("📊 نمایش آمار"):
        st.session_state.show_analytics = True
    
    if st.sidebar.button("📜 تاریخچه"):
        st.session_state.show_history = True
    
    if st.sidebar.button("❌ پاک کردن تاریخچه"):
        st.session_state.query_history = []
        st.session_state.processing_times = []
        st.success("تاریخچه پاک شد!")
    
    st.sidebar.markdown("---")
    
    # Help and information
    st.sidebar.markdown("### ❓ راهنما")
    
    with st.sidebar.expander("💡 نمونه سوالات"):
        st.markdown("""
        - ایمیل استاد جهانگرد چیست؟
        - برنامه کلاسی استاد محمدی را نشان بده
        - سوابق تحصیلی استاد احمدی چیست؟
        - اساتید گروه کامپیوتر کدامند؟
        - استاد دانشیار حقوق چه کسی است؟
        """)
    
    with st.sidebar.expander("🔍 روش‌های جستجو"):
        methods = get_retrieval_methods()
        if methods:
            for method, info in methods.get('retrieval_methods', {}).items():
                st.markdown(f"**{method.title()}:**")
                st.markdown(f"_{info['description']}_")
        else:
            st.info("اطلاعات در دسترس نیست")
    
    st.sidebar.markdown("---")
    
    # Footer
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 20px 0; color: #7f8c8d;">
            <p style="font-size: 12px; margin: 0;">
                نسخه {version}<br>
                دانشگاه یزد
            </p>
        </div>
        """.format(version=APP_CONFIG["version"]),
        unsafe_allow_html=True
    )

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    if st.session_state.get('show_analytics', False):
        render_analytics()
        if st.button("🔙 بازگشت"):
            st.session_state.show_analytics = False
            st.rerun()
        return
    
    if st.session_state.get('show_history', False):
        render_query_history()
        if st.button("🔙 بازگشت"):
            st.session_state.show_history = False
            st.rerun()
        return
    
    # Check backend status
    if not st.session_state.backend_status:
        st.error(
            """
            ⚠️ **خطا در اتصال به سرور**
            
            سرور در دسترس نیست. لطفاً موارد زیر را بررسی کنید:
            - سرور backend در حال اجرا است
            - آدرس سرور صحیح است
            - اتصال اینترنت برقرار است
            """
        )
        return
    
    # Query form
    query, use_self_query, submit_button = render_query_form()
    
    # Process query
    if submit_button and query:
        with st.spinner("🔄 در حال پردازش پرسش..."):
            start_time = time.time()
            
            # Send query to backend
            response_data = send_query(query, use_self_query)
            
            if response_data:
                # Record processing time
                processing_time = time.time() - start_time
                st.session_state.processing_times.append(processing_time)
                
                # Add to history
                st.session_state.query_history.append((
                    query,
                    response_data,
                    datetime.now()
                ))
                
                # Render response
                render_response(response_data)
                
                # Show success message
                st.success(f"✅ پرسش با موفقیت پردازش شد! (زمان: {format_processing_time(processing_time)})")
            else:
                st.error("❌ خطا در پردازش پرسش. لطفاً دوباره تلاش کنید.")
    
    # Show query history if available
    if st.session_state.query_history:
        st.markdown("---")
        render_query_history()

if __name__ == "__main__":
    main() 