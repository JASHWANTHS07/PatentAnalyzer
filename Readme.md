# Patent Trend Analyzer

## Overview

Patent Trend Analyzer is a powerful Streamlit-based web application that helps researchers, inventors, and businesses analyze patent trends in specific industries and technology areas. The tool fetches patent data from Google Patents, performs in-depth analysis, and generates visualizations and insights to help users understand innovation landscapes.

## Features

-   **Patent Discovery**: Automatically fetches relevant patents from Google Patents based on industry and topic keywords
-   **Trend Analysis**: Identifies emerging trends and patterns in patent filings
-   **Interactive Visualizations**: Provides multiple visualization types including:
    -   Patent publication timeline
    -   Assignee distribution
    -   Term frequency word clouds
    -   Technology clusters
    -   Innovation trend analysis
-   **Comprehensive Summaries**: Generates detailed summaries of patent analysis findings
-   **Downloadable Reports**: Creates markdown reports that can be downloaded for offline use
-   **User-friendly Interface**: Simple, intuitive interface with basic and advanced analysis modes

## Installation

### Prerequisites

-   Python 3.7+
-   pip (Python package manager)

### Dependencies

The application requires several Python libraries:

```
streamlit
requests
faiss-cpu
numpy
pandas
sentence-transformers
huggingface_hub
langchain
langchain_community
matplotlib
seaborn
wordcloud
scikit-learn
```

### Setup

1. Install the required packages:

```bash
pip install faiss-cpu streamlit requests numpy seaborn scikit-learn Wordcloud huggingface_hub sentence-transformers matplotlib langchain_community
```

2. API Keys:

    - **_You will need api keys for this code to run completely._**
    - The application uses SerpAPI for patent data retrieval. You need to obtain an API key from [SerpAPI](https://serpapi.com/).
    - For enhanced summarization capabilities, a Hugging Face API token is recommended.

3. (Optional) Set up environment variables or Streamlit secrets:

    - Create a `.streamlit/secrets.toml` file in your project directory:

    ```toml
    [serp_api]
    api_key = "your_serpapi_key_here"

    HF_TOKEN = "your_huggingface_token_here"
    ```

### Make sure all mentioned Libraries are installed and you are capable of running Streamlit (difficult to run streamlit on Google Colab).

## Usage

### Running the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

This will launch the application in your default web browser.

### Quick Analysis Mode

1. Enter your industry of interest (e.g., "Mobile Phone Manufacturing")
2. Specify a technology topic (e.g., "Battery Technology", "Foldable Displays")
3. Click "Analyze Patent Trends"
4. View the results and download the report if desired

### Advanced Analysis Mode

1. Enter industry and topic information
2. Configure additional parameters:
    - Minimum publication year
    - Number of patents to analyze
    - Visualization options
3. Click "Run Advanced Analysis"
4. Explore the detailed results and visualizations

## Technical Architecture

The application consists of several key components:

### 1. PatentFetcher

Responsible for retrieving patent data from Google Patents via SerpAPI. Constructs search queries based on user input and processes the API response.

### 2. TrendAnalyzer

Processes patent data using sentence transformers and FAISS for vector similarity search. Identifies patterns and relationships between patents.

### 3. PatentVisualizer

Creates various visualizations to help users understand patent trends:

-   Timeline charts showing patent publication activity over time
-   Assignee distribution charts identifying leading organizations
-   Word clouds highlighting key terms
-   Technology cluster visualizations using dimensionality reduction
-   Innovation trend charts tracking terminology over time

### 4. LLMSummarizer

Generates comprehensive summaries of patent analysis findings, including key trends and recommendations.

## Troubleshooting

### API Key Issues

If you encounter errors related to API requests:

-   Verify your SerpAPI key is correct
-   Check your internet connection
-   Ensure you haven't exceeded API usage limits

### Visualization Errors

If visualizations fail to render:

-   Check that you have sufficient patent data (some visualizations require multiple patents)
-   Ensure all dependencies are properly installed

### Missing Secrets Warning

If you see a "No secrets found" warning:

-   Create a `.streamlit/secrets.toml` file as described in the setup section
-   Alternatively, modify the code to use environment variables or direct values

## Extending the Application

### Adding New Visualizations

The modular design makes it easy to add new visualization types:

1. Create a new method in the `PatentVisualizer` class
2. Add the visualization type to the results dictionary in `generate_all_visualizations`
3. Update the UI to display the new visualization

### Enhancing Analysis Capabilities

To add new analysis features:

1. Create a new analysis method or class
2. Integrate it into the `execute_patent_analysis` function
3. Update the UI to display the new analysis results

## Demo

![Patent Trend Analyzer Demo](./PA Demo.mp4)

## License

[MIT License](LICENSE)

## Acknowledgments

-   This application uses data from Google Patents via SerpAPI
-   Visualization components built with Matplotlib and Seaborn
-   Text analysis powered by Sentence Transformers and scikit-learn
