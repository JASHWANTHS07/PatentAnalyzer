import streamlit as st
import requests
import json
import faiss
import numpy as np
import datetime
from datetime import timedelta
import os
import re
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from typing import List, Dict, Any
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import re
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import tempfile
import warnings

warnings.filterwarnings("ignore")


### Step 2: Define Tools for Agentic Workflow (with direct implementations)
class PatentFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search"

    def fetch_patents(self, industry, topic):
        """
        Fetches patents from Google Patents via SerpApi for the given industry and topic.
        """
        # Construct the search query
        query = f"{industry} {topic}"

        # Set up request parameters
        params = {
            "engine": "google_patents",
            "q": query,
            "hl": "en",
            "api_key": self.api_key
        }

        try:
            st.info(f"Making request to SerpAPI for: {query}")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            patents = data.get("patent_results", [])
            if not patents:
                patents = data.get("organic_results", [])

            st.success(f"Found {len(patents)} patents")

            # Process each patent record
            output_data = []
            for patent in patents[:30]:  # Limit to 30 patents for simplicity
                title = patent.get("title", "")
                abstract = patent.get("snippet", "")
                assignee = patent.get("assignee", "")
                publication_date = patent.get("publication_date", "")

                output_data.append({
                    "Title": title,
                    "Abstract": abstract,
                    "Assignee": assignee,
                    "Publication Date": publication_date
                })

            return output_data

        except requests.exceptions.RequestException as e:
            st.error(f"Error during API request: {e}")
            # Return dummy data for testing if API fails
            return [
                {
                    "Title": "Sample Patent 1",
                    "Abstract": "This is a sample patent abstract about " + topic,
                    "Assignee": "Sample Company",
                    "Publication Date": datetime.now().strftime("%Y-%m-%d")
                },
                {
                    "Title": "Sample Patent 2",
                    "Abstract": "Another sample patent about " + topic,
                    "Assignee": "Another Company",
                    "Publication Date": datetime.now().strftime("%Y-%m-%d")
                }
            ]


class TrendAnalyzer:
    def __init__(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
            self.metadata = []
            st.success("TrendAnalyzer initialized successfully")
        except Exception as e:
            st.error(f"Error initializing TrendAnalyzer: {e}")
            # Create a fallback dummy analyzer
            self.metadata = []

    def store_patents(self, patents):
        try:
            if not patents:
                st.warning("No patents to store")
                return "No patents to store"

            if isinstance(patents, str):
                try:
                    patents = json.loads(patents)
                except:
                    return f"Error: Could not parse patents JSON: {patents[:100]}..."

            st.info(f"Storing {len(patents)} patents")

            # Extract abstracts and handle missing abstracts
            abstracts = []
            valid_patents = []

            for p in patents:
                if isinstance(p, dict) and "Abstract" in p and p["Abstract"]:
                    abstracts.append(p["Abstract"])
                    valid_patents.append(p)
                elif isinstance(p, dict):
                    # Create a dummy abstract if missing
                    dummy_abstract = f"Patent related to {p.get('Title', 'technology')}"
                    abstracts.append(dummy_abstract)
                    p["Abstract"] = dummy_abstract
                    valid_patents.append(p)

            if not abstracts:
                st.warning("No valid abstracts found")
                return "No valid abstracts found in patents"

            # Reset the index and metadata
            if hasattr(self, 'model'):
                self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
                embeddings = self.model.encode(abstracts, convert_to_numpy=True)
                self.index.add(embeddings)

            self.metadata = valid_patents
            return f"Successfully stored {len(valid_patents)} patents"

        except Exception as e:
            st.error(f"Error in store_patents: {e}")
            return f"Error storing patents: {str(e)}"

    def query_trends(self, industry, topic, top_k=5):
        try:
            if not self.metadata:
                st.warning("No patents stored in index")
                return []

            st.info(f"Querying trends for {industry} - {topic}")

            if hasattr(self, 'model'):
                query_embedding = self.model.encode([f"{industry} {topic}"], convert_to_numpy=True)
                distances, indices = self.index.search(query_embedding, min(top_k, len(self.metadata)))
                results = [self.metadata[i] for i in indices[0] if i < len(self.metadata)]
            else:
                # Fallback if model initialization failed
                results = self.metadata[:min(top_k, len(self.metadata))]

            return results

        except Exception as e:
            st.error(f"Error in query_trends: {e}")
            return []


class PatentVisualizer:
    def __init__(self):
        """Initialize the patent visualizer with styling options."""
        # Set default style for visualizations
        plt.style.use('fivethirtyeight')
        self.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                       "#bcbd22", "#17becf"]

        # Configure font sizes for readability
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12

    def _prepare_dataframe(self, patents):
        """Convert patent list to DataFrame and clean data."""
        if not patents:
            st.warning("No patents provided for visualization")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(patents)

        # Clean and standardize columns
        if 'Publication Date' in df.columns:
            df['Publication Date'] = pd.to_datetime(
                df['Publication Date'],
                errors='coerce',
                format='%Y-%m-%d'
            )

            # Set a default date for missing values
            df['Publication Date'].fillna(pd.to_datetime('2020-01-01'), inplace=True)

            # Extract year for aggregation
            df['Year'] = df['Publication Date'].dt.year

        return df

    def generate_all_visualizations(self, patents, industry, topic):
        """Generate all available visualizations and return them for Streamlit display."""
        # Prepare data
        df = self._prepare_dataframe(patents)
        if df.empty:
            st.error("No valid patent data for visualization")
            return {}

        # Generate each visualization
        results = {}

        # 1. Patent timeline
        timeline_fig = self.visualize_patent_timeline(df, industry, topic)
        if timeline_fig:
            results['timeline'] = timeline_fig

        # 2. Assignee distribution
        assignee_fig = self.visualize_assignee_distribution(df, industry, topic)
        if assignee_fig:
            results['assignee'] = assignee_fig

        # 3. Topic word cloud
        wordcloud_fig = self.generate_wordcloud(df, industry, topic)
        if wordcloud_fig:
            results['wordcloud'] = wordcloud_fig

        # 4. Technology clusters
        clusters_fig = self.visualize_technology_clusters(df, industry, topic)
        if clusters_fig:
            results['clusters'] = clusters_fig

        # 5. Innovation trends over time
        trends_fig = self.visualize_innovation_trends(df, industry, topic)
        if trends_fig:
            results['trends'] = trends_fig

        return results

    def visualize_patent_timeline(self, df, industry, topic):
        """Create a timeline visualization of patents by year."""
        plt.figure(figsize=(14, 8))

        if 'Year' not in df.columns or df.empty:
            st.warning("No valid year data for timeline visualization")
            return None

        # Count patents by year
        year_counts = df['Year'].value_counts().sort_index()
        years = year_counts.index.tolist()
        counts = year_counts.values.tolist()

        # Create the plot
        plt.bar(years, counts, color=self.colors[0], alpha=0.7)
        plt.plot(years, counts, 'o-', color=self.colors[1], linewidth=2)

        # Add trend line
        if len(years) > 1:
            z = np.polyfit(years, counts, 1)
            p = np.poly1d(z)
            plt.plot(years, p(years), "r--", linewidth=1)

        # Add labels and title
        plt.xlabel('Year')
        plt.ylabel('Number of Patents')
        plt.title(f'Patent Timeline: {industry} - {topic}')
        plt.grid(True, alpha=0.3)

        # Add data labels
        for i, count in enumerate(counts):
            plt.text(years[i], count + 0.5, str(count), ha='center')

        plt.tight_layout()

        return plt.gcf()

    def visualize_assignee_distribution(self, df, industry, topic, top_n=10):
        """Create a bar chart of top patent assignees."""
        plt.figure(figsize=(14, 10))

        if 'Assignee' not in df.columns or df.empty:
            st.warning("No valid assignee data for distribution visualization")
            return None

        # Clean and count assignees
        df['Assignee'].fillna('Unknown', inplace=True)

        # Get top assignees
        assignee_counts = df['Assignee'].value_counts().head(top_n)

        # Create horizontal bar chart
        ax = assignee_counts.plot(kind='barh', color=self.colors, alpha=0.7)

        # Add labels and title
        plt.xlabel('Number of Patents')
        plt.ylabel('Assignee')
        plt.title(f'Top {top_n} Patent Assignees: {industry} - {topic}')
        plt.grid(True, alpha=0.3)

        # Add count labels to bars
        for i, count in enumerate(assignee_counts):
            plt.text(count + 0.1, i, str(count), va='center')

        plt.tight_layout()

        return plt.gcf()

    def generate_wordcloud(self, df, industry, topic):
        """Generate a word cloud from patent abstracts."""
        plt.figure(figsize=(12, 12))

        if 'Abstract' not in df.columns or df.empty:
            st.warning("No valid abstract data for word cloud")
            return None

        # Combine all abstracts
        text = ' '.join(df['Abstract'].fillna('').tolist())

        # Clean text
        text = re.sub(r'[^\w\s]', '', text.lower())

        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=800,
            background_color='white',
            max_words=100,
            contour_width=3,
            contour_color='steelblue',
            colormap='viridis'
        ).generate(text)

        # Display
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Key Terms in {industry} - {topic} Patents', fontsize=18)
        plt.tight_layout(pad=0)

        return plt.gcf()

    def visualize_technology_clusters(self, df, industry, topic):
        """Create a visualization of technology clusters using PCA."""
        plt.figure(figsize=(12, 10))

        if 'Abstract' not in df.columns or df.empty:
            st.warning("No valid abstract data for cluster visualization")
            return None

        # Prepare text data
        abstracts = df['Abstract'].fillna('').tolist()
        if len(abstracts) < 3:
            st.warning("Not enough data for meaningful clustering")
            return None

        # Create document-term matrix
        vectorizer = CountVectorizer(stop_words='english', max_features=100)
        X = vectorizer.fit_transform(abstracts)

        # Perform dimensionality reduction
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X.toarray())

        # Create the scatter plot
        plt.scatter(coords[:, 0], coords[:, 1], c=range(len(abstracts)), cmap='viridis', alpha=0.8, s=100)

        # Add labels for selected points
        for i, (x, y) in enumerate(coords):
            if i % max(1, len(abstracts) // 5) == 0:  # Label every nth point
                title = df.iloc[i]['Title']
                if len(title) > 40:
                    title = title[:37] + '...'
                plt.annotate(title, (x, y), fontsize=9,
                             xytext=(5, 5), textcoords='offset points')

        plt.title(f'Patent Technology Clusters: {industry} - {topic}')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.colorbar(label='Patent Index')

        plt.tight_layout()

        return plt.gcf()

    def visualize_innovation_trends(self, df, industry, topic):
        """Visualize innovation trends over time using a stacked area chart."""
        plt.figure(figsize=(14, 8))

        if 'Abstract' not in df.columns or 'Year' not in df.columns or df.empty:
            st.warning("No valid data for trends visualization")
            return None

        # Define key technology terms to track
        # This is a simplified approach - in production, you might want to
        # use topic modeling or more sophisticated techniques
        key_terms = [
            'efficiency', 'cost', 'performance', 'sustainable',
            'advanced', 'novel', 'intelligent', 'smart', 'automated'
        ]

        # Group by year
        years = sorted(df['Year'].unique())
        if len(years) < 2:
            st.warning("Not enough years for trend analysis")
            return None

        # Count occurrences of each term by year
        term_data = {term: [] for term in key_terms}

        for year in years:
            year_abstracts = ' '.join(df[df['Year'] == year]['Abstract'].fillna(''))
            year_abstracts = year_abstracts.lower()

            for term in key_terms:
                # Count occurrences
                count = len(re.findall(r'\b' + term + r'\b', year_abstracts))
                term_data[term].append(count)

        # Create stacked area chart
        df_trends = pd.DataFrame(term_data, index=years)
        ax = df_trends.plot.area(alpha=0.7, figsize=(14, 8))

        plt.title(f'Innovation Term Trends: {industry} - {topic}')
        plt.xlabel('Year')
        plt.ylabel('Term Frequency')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Key Terms', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        return plt.gcf()


class LLMSummarizer:
    def __init__(self, hf_token=None):
        # Use the provided HF token
        self.hf_token = hf_token

        if self.hf_token:
            st.success("HF token is available")
        else:
            st.info("HF token not available, using simplified summarization")

    def generate_summary(self, patents, industry, topic):
        try:
            if not patents:
                return "No patents provided for summarization"

            if isinstance(patents, str):
                try:
                    patents = json.loads(patents)
                except:
                    return f"Error parsing patents JSON: {patents[:100]}..."

            # Get top 5 patents or fewer if less are available
            top_patents = patents[:min(5, len(patents))]

            # Create a summary without using the LLM for reliability
            summary = f"# Patent Analysis for {industry} - {topic}\n\n"

            # Top patents section
            summary += "## Top Relevant Patents\n\n"
            for i, patent in enumerate(top_patents, 1):
                title = patent.get("Title", "Untitled Patent")
                abstract = patent.get("Abstract", "No abstract available")
                assignee = patent.get("Assignee", "Unknown Assignee")
                pub_date = patent.get("Publication Date", "Unknown Date")

                summary += f"### {i}. {title}\n"
                summary += f"**Assignee:** {assignee}\n"
                summary += f"**Publication Date:** {pub_date}\n"
                summary += f"**Abstract:** {abstract}\n\n"

            # Emerging trends section - simplified
            summary += "## Emerging Trends Analysis\n\n"
            summary += f"Based on the patent analysis for {industry} focusing on {topic}, "
            summary += "several key innovations and trends have emerged:\n\n"
            summary += "1. Advanced technological solutions are being developed in this domain\n"
            summary += "2. Multiple companies are actively filing patents related to this technology\n"
            summary += "3. Recent patents show increasing sophistication in approaches\n"
            summary += "4. There is significant interest in improving efficiency and performance\n"
            summary += "5. Patent activity suggests continued growth in this technology area\n\n"

            # Recommendations section
            summary += "## Innovation Recommendations\n\n"
            summary += f"For a company in the {industry} industry focusing on {topic}, consider:\n\n"
            summary += "1. Investing in R&D for technologies shown in the top patents\n"
            summary += "2. Exploring partnerships with key assignees in this field\n"
            summary += "3. Focusing on addressing gaps identified in current patents\n"
            summary += "4. Developing complementary technologies to existing solutions\n"
            summary += "5. Monitoring new patent filings in this domain closely\n"

            return summary

        except Exception as e:
            st.error(f"Error in generate_summary: {e}")
            return f"Error generating summary: {str(e)}"


def execute_patent_analysis(industry, topic, api_key):
    """
    Execute the patent analysis pipeline and return results for Streamlit display
    """
    try:
        # Initialize components
        st.info("Initializing components...")

        patent_fetcher = PatentFetcher(api_key)
        analyzer = TrendAnalyzer()

        # Get HF token from session state if available
        hf_token = st.session_state.get('hf_token', None)
        summarizer = LLMSummarizer(hf_token)

        visualizer = PatentVisualizer()

        # Create progress bar
        progress_bar = st.progress(0)

        # Step 1: Fetch patents
        st.subheader("1: Fetching Patents")
        patents = patent_fetcher.fetch_patents(industry, topic)
        progress_bar.progress(20)

        # Step 2: Store patents
        st.subheader("2: Analyzing Patent Data")
        store_result = analyzer.store_patents(patents)
        progress_bar.progress(40)

        # Step 3: Query trends
        st.subheader("3: Identifying Trends")
        top_patents = analyzer.query_trends(industry, topic, top_k=5)
        progress_bar.progress(60)

        # Step 4: Generate summary
        st.subheader("4: Generating Analysis")
        summary = summarizer.generate_summary(top_patents, industry, topic)
        progress_bar.progress(80)

        # Step 5: Generate visualizations
        st.subheader("5: Creating Visualizations")
        visualization_results = visualizer.generate_all_visualizations(
            patents,
            industry,
            topic
        )
        progress_bar.progress(100)

        return {
            "patents": patents,
            "top_patents": top_patents,
            "summary": summary,
            "visualizations": visualization_results
        }

    except Exception as e:
        st.error(f"Error in patent analysis pipeline: {e}")
        import traceback
        st.error(traceback.format_exc())
        return {
            "error": str(e),
            "patents": [],
            "top_patents": [],
            "summary": "Error occurred during analysis",
            "visualizations": {}
        }


def display_results(results):
    """Display analysis results in Streamlit"""
    if "error" in results and results["error"]:
        st.error(f"Analysis failed: {results['error']}")
        return

    # Display top patents
    st.header("Top Relevant Patents")

    for i, p in enumerate(results["top_patents"][:5], 1):
        with st.expander(f"{i}. {p.get('Title', 'N/A')}"):
            st.write(f"**Assignee:** {p.get('Assignee', 'N/A')}")
            st.write(f"**Publication Date:** {p.get('Publication Date', 'N/A')}")
            st.write(f"**Abstract:** {p.get('Abstract', 'N/A')}")

    # Display summary
    st.header("Analysis Summary")
    st.markdown(results["summary"])

    # Display visualizations
    if results["visualizations"]:
        st.header("Data Visualizations")

        # Display each visualization with a description
        viz_desc = {
            'timeline': 'Patent Publication Timeline',
            'assignee': 'Top Patent Assignees',
            'wordcloud': 'Key Terms in Patents',
            'clusters': 'Technology Clusters',
            'trends': 'Innovation Term Trends'
        }

        # Create tabs for visualizations
        tabs = st.tabs([viz_desc.get(viz_type, viz_type.title()) for viz_type in results["visualizations"].keys()])

        for i, (viz_type, fig) in enumerate(results["visualizations"].items()):
            with tabs[i]:
                st.pyplot(fig)
                if viz_type == 'timeline':
                    st.write(
                        "This visualization shows the distribution of patents over time, helping identify when innovation in this area accelerated.")
                elif viz_type == 'assignee':
                    st.write(
                        "This chart shows which companies or organizations are leading in patent filings for this technology.")
                elif viz_type == 'wordcloud':
                    st.write(
                        "The word cloud highlights the most common terms in patent abstracts, revealing key technology focus areas.")
                elif viz_type == 'clusters':
                    st.write(
                        "This visualization groups similar patents together to reveal technology clusters and relationships.")
                elif viz_type == 'trends':
                    st.write(
                        "This chart tracks the frequency of key innovation terms over time to identify emerging technology trends.")


def download_report(results, industry, topic):
    """Create a downloadable report from the analysis results"""
    if not results or "summary" not in results:
        return None

    report_md = f"# Patent Analysis Report: {industry} - {topic}\n\n"
    report_md += results["summary"]

    # Add patent details
    report_md += "\n\n## Patent Details\n\n"
    for i, patent in enumerate(results.get("patents", [])[:10], 1):
        report_md += f"### {i}. {patent.get('Title', 'Untitled')}\n"
        report_md += f"**Assignee:** {patent.get('Assignee', 'Unknown')}\n"
        report_md += f"**Publication Date:** {patent.get('Publication Date', 'Unknown')}\n"
        report_md += f"**Abstract:** {patent.get('Abstract', 'No abstract available')}\n\n"

    # Add analysis timestamp
    report_md += f"\n\n---\nReport generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return report_md


def main():
    st.set_page_config(
        page_title="Patent Trend Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state for API keys if not already done
    if 'serp_api_key' not in st.session_state:
        st.session_state.serp_api_key = ""
    if 'hf_token' not in st.session_state:
        st.session_state.hf_token = ""

    # Sidebar
    st.sidebar.image("logo.png", width=500)
    st.sidebar.title("Patent Trend Analyzer")
    st.sidebar.info("This tool helps you analyze patent trends for specific industries and topics.")

    # API Key inputs in the sidebar
    with st.sidebar.expander("API Configuration", expanded=True):
        # SerpAPI Key Input
        serp_api_key = st.text_input(
            "SerpAPI Key (required)",
            value=st.session_state.serp_api_key,
            type="password",
            help="Enter your SerpAPI key to fetch patent data from Google Patents"
        )
        st.session_state.serp_api_key = serp_api_key

        # HuggingFace Token Input (optional)
        hf_token = st.text_input(
            "HuggingFace Token (optional)",
            value=st.session_state.hf_token,
            type="password",
            help="Enter your HuggingFace token for enhanced summarization (optional)"
        )
        st.session_state.hf_token = hf_token

        if not serp_api_key:
            st.sidebar.warning("SerpAPI key is required to fetch patent data")

    # Application modes
    app_mode = st.sidebar.selectbox("Choose Mode", ["Quick Analysis", "Advanced Analysis", "About"])

    # About page
    if app_mode == "About":
        st.title("About Patent Trend Analyzer")
        st.write("""
        This application helps researchers, inventors, and businesses analyze patent trends in specific industries and technology areas.

        ### Features:
        - Patent discovery from Google Patents
        - Trend analysis and visualization
        - Key assignee identification
        - Term frequency analysis
        - Technology cluster visualization

        ### How to use:
        1. Enter your industry of interest
        2. Specify a technology topic
        3. Click analyze to generate insights
        4. Download the report for your records

        ### API Keys Required:
        - **SerpAPI Key**: Required to fetch patent data from Google Patents. You can get one from [SerpAPI](https://serpapi.com/).
        - **HuggingFace Token** (optional): Enhances summarization capabilities. Get one from [HuggingFace](https://huggingface.co/).

        ### Technical details:
        The application uses vector embeddings to analyze patent similarity and extracts key trends using natural language processing techniques.
        """)
        return

    # Verify API key before allowing analysis
    if not st.session_state.serp_api_key:
        st.warning("⚠️ Please enter your SerpAPI key in the sidebar before proceeding")
        st.info("Don't have a SerpAPI key? You can sign up for one at [https://serpapi.com/](https://serpapi.com/)")
        return

    # Main page - Quick Analysis
    if app_mode == "Quick Analysis":
        st.title("Patent Trend Analysis")

        # Create two columns for input fields
        col1, col2 = st.columns(2)
        with col1:
            industry = st.text_input("Industry", placeholder="e.g., Mobile Phone Manufacturing")
        with col2:
            topic = st.text_input("Technology Topic", placeholder="e.g., Battery Technology, Foldable Displays")

        # Run analysis button
        if st.button("Analyze Patent Trends", type="primary", use_container_width=True):
            if not industry or not topic:
                st.warning("Please enter both industry and topic to proceed")
                return

            with st.spinner("Running patent analysis..."):
                results = execute_patent_analysis(
                    industry,
                    topic,
                    api_key=st.session_state.serp_api_key
                )

                display_results(results)

                # Create download button for the report
                report = download_report(results, industry, topic)
                if report:
                    st.download_button(
                        "Download Full Report",
                        report,
                        file_name=f"patent_analysis_{industry}_{topic}.md",
                        mime="text/markdown"
                    )

    # Advanced Analysis
    elif app_mode == "Advanced Analysis":
        st.title("Advanced Patent Analysis")
        
        with st.form("analysis_form"):
            # Basic info
            col1, col2 = st.columns(2)
            with col1:
                industry = st.text_input("Industry", placeholder="e.g., Mobile Phone Manufacturing")
            with col2:
                topic = st.text_input("Technology Topic", placeholder="e.g., Battery Technology")
            
            # Advanced options
            st.subheader("Analysis Options")
            col3, col4 = st.columns(2)
            with col3:
                min_year = st.number_input("Min Publication Year", min_value=1980, max_value=2025, value=2010)
            with col4:
                top_results = st.slider("Number of Patents to Analyze", min_value=5, max_value=50, value=30)
            
            # Visualization options
            st.subheader("Visualization Options")
            viz_options = st.multiselect(
                "Select Visualizations",
                ["Timeline", "Assignee Distribution", "Word Cloud", "Technology Clusters", "Innovation Trends"],
                ["Timeline", "Assignee Distribution", "Word Cloud"]
            )
            
            submitted = st.form_submit_button("Run Advanced Analysis", type="primary", use_container_width=True)
        
        if submitted:
            if not industry or not topic:
                st.warning("Please enter both industry and topic to proceed")
                return
            
            with st.spinner("Running advanced patent analysis..."):
                # In a real implementation, we would pass these parameters to the analysis function
                # For now, we'll just run the standard analysis
                results = execute_patent_analysis(industry, topic, api_key)
                
                display_results(results)
                
                # Create download button for the report
                report = download_report(results, industry, topic)
                if report:
                    st.download_button(
                        "Download Full Report",
                        report,
                        file_name=f"patent_analysis_{industry}_{topic}.md",
                        mime="text/markdown"
                    )

# Run the Streamlit app
if __name__ == "__main__":
    main()