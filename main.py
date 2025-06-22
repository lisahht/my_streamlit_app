"""
Zimbabwean News Spider and Clustering Platform
==============================================

This script crawls Zimbabwean newspapers, extracts articles by category,
performs clustering analysis, and provides a Streamlit web interface.

Usage:
    streamlit run main.py

Features:
- Dynamic URL structure discovery
- Multi-threaded web scraping
- Text clustering using TF-IDF and K-means
- Interactive Streamlit dashboard
- CSV data export
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from urllib.parse import urljoin, urlparse
import re
import time
from queue import Queue
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NewsSpider:
    def __init__(self):
        self.base_urls = [
            'http://www.herald.co.zw/',
            'https://www.newsday.co.zw/',
            'https://www.theindependent.co.zw/',
            'https://dailynews.co.zw/'
        ]
        
        self.target_categories = {
            'business': ['business', 'economy', 'finance', 'economic', 'market'],
            'politics': ['politics', 'political', 'government', 'parliament', 'policy'],
            'arts_culture': ['arts', 'culture', 'entertainment', 'celebrity', 'music', 'film', 'theatre'],
            'sports': ['sport', 'football', 'cricket', 'rugby', 'soccer', 'athletics']
        }
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.articles_data = []
        self.url_structures = {}

    def discover_url_structures(self):
        """Discover URL structures for each newspaper website"""
        st.info("🔍 Discovering URL structures for each newspaper...")
        
        for base_url in self.base_urls:
            try:
                st.write(f"Analyzing: {base_url}")
                structure = self._analyze_site_structure(base_url)
                self.url_structures[base_url] = structure
                st.success(f"✅ Found {len(structure)} category URLs for {base_url}")
            except Exception as e:
                st.error(f"❌ Failed to analyze {base_url}: {str(e)}")
                self.url_structures[base_url] = {}

    def _analyze_site_structure(self, base_url):
        """Analyze a single site to find category URLs"""
        try:
            response = self.session.get(base_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links on the homepage
            links = soup.find_all('a', href=True)
            category_urls = {}
            
            for link in links:
                href = link.get('href')
                text = link.get_text().strip().lower()
                
                # Convert relative URLs to absolute
                if href:
                    full_url = urljoin(base_url, href)
                    
                    # Check if this link matches any of our target categories
                    for category, keywords in self.target_categories.items():
                        if any(keyword in text or keyword in href.lower() for keyword in keywords):
                            if category not in category_urls:
                                category_urls[category] = []
                            if full_url not in category_urls[category]:
                                category_urls[category].append(full_url)
            
            return category_urls
            
        except Exception as e:
            print(f"Error analyzing {base_url}: {e}")
            return {}

    def scrape_articles(self, max_articles_per_category=10):
        """Scrape articles from discovered URLs"""
        st.info("📰 Scraping articles from discovered category URLs...")
        
        progress_bar = st.progress(0)
        total_operations = sum(len(urls) for urls in self.url_structures.values() if urls)
        current_operation = 0
        
        for base_url, categories in self.url_structures.items():
            site_name = urlparse(base_url).netloc
            st.write(f"Scraping from: {site_name}")
            
            for category, urls in categories.items():
                for url in urls[:2]:  
                    try:
                        articles = self._scrape_category_page(url, category, site_name, max_articles_per_category)
                        self.articles_data.extend(articles)
                        
                        current_operation += 1
                        progress_bar.progress(current_operation / total_operations)
                        
                        time.sleep(1)  # Be respectful to the servers
                        
                    except Exception as e:
                        st.warning(f"⚠️ Error scraping {url}: {str(e)}")
                        
        st.success(f"✅ Successfully scraped {len(self.articles_data)} articles")

    def _scrape_category_page(self, url, category, site_name, max_articles=10):
        """Scrape articles from a category page"""
        articles = []
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            article_links = []
            
            # Pattern 1: Links in article titles, headings
            for tag in soup.find_all(['h1', 'h2', 'h3', 'h4']):
                link = tag.find('a', href=True)
                if link:
                    article_links.append(link)
            
            # Pattern 2: Links with article-related classes
            for class_name in ['article', 'post', 'story', 'news-item', 'entry']:
                links = soup.find_all('a', class_=re.compile(class_name, re.I))
                article_links.extend(links)
            
            # Pattern 3: Links in common article containers
            for container in soup.find_all(['article', 'div'], class_=re.compile(r'(article|post|story|news)', re.I)):
                links = container.find_all('a', href=True)
                article_links.extend(links)
            
            # Process found links
            processed_urls = set()
            for link in article_links[:max_articles * 2]:  # Get extra in case some fail
                href = link.get('href')
                title = link.get_text().strip()
                
                if href and title and len(title) > 10:  # Filter out short/empty titles
                    full_url = urljoin(url, href)
                    
                    if full_url not in processed_urls and self._is_article_url(full_url):
                        processed_urls.add(full_url)
                        
                        # Try to scrape the individual article
                        article_data = self._scrape_individual_article(full_url, title, category, site_name)
                        if article_data:
                            articles.append(article_data)
                            
                        if len(articles) >= max_articles:
                            break
                            
        except Exception as e:
            print(f"Error scraping category page {url}: {e}")
            
        return articles

    def _is_article_url(self, url):
        """Check if URL looks like an individual article"""
        # Avoid category pages, home pages, etc.
        avoid_patterns = ['/category/', '/tag/', '/page/', '/archive/', '#', 'javascript:']
        return not any(pattern in url.lower() for pattern in avoid_patterns)

    def _scrape_individual_article(self, url, title, category, site_name):
        """Scrape content from an individual article"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article content using multiple strategies
            content = self._extract_article_content(soup)
            
            if len(content) > 100:  # Only include articles with substantial content
                return {
                    'title': title,
                    'content': content,
                    'category': category,
                    'site': site_name,
                    'url': url,
                    'scraped_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"Error scraping article {url}: {e}")
            
        return None

    def _extract_article_content(self, soup):
        """Extract main article content from HTML"""
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Try multiple content extraction strategies
        content = ""
        
        # Strategy 1: Look for common article content containers
        for selector in [
            'article', '[class*="content"]', '[class*="article"]', 
            '[class*="post"]', '[class*="entry"]', '.story-body', 
            '.article-body', '.post-content', '.entry-content'
        ]:
            element = soup.select_one(selector)
            if element:
                content = element.get_text(strip=True)
                if len(content) > 200:
                    break
        
        # Strategy 2: Look for paragraphs in main content area
        if len(content) < 200:
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text(strip=True) for p in paragraphs[:10]])
        
        # Clean up the content
        content = re.sub(r'\s+', ' ', content)
        return content[:2000]  # Limit content length

    def save_to_csv(self, filename='scraped_articles.csv'):
        """Save scraped articles to CSV file"""
        if not self.articles_data:
            st.warning("⚠️ No articles to save!")
            return False
            
        try:
            df = pd.DataFrame(self.articles_data)
            df.to_csv(filename, index=False)
            st.success(f"✅ Saved {len(self.articles_data)} articles to {filename}")
            return True
        except Exception as e:
            st.error(f"❌ Error saving to CSV: {str(e)}")
            return False

    def perform_clustering(self, n_clusters=4):
        """Perform K-means clustering on article content"""
        if len(self.articles_data) < n_clusters:
            st.error(f"❌ Not enough articles ({len(self.articles_data)}) for {n_clusters} clusters")
            return None
            
        st.info("🔬 Performing text clustering analysis...")
        
        # Prepare text data
        texts = [article['content'] for article in self.articles_data]
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Add cluster labels to articles data
            for i, article in enumerate(self.articles_data):
                article['cluster'] = int(cluster_labels[i])
            
            # PCA for visualization
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(tfidf_matrix.toarray())
            
            # Create visualization dataframe
            viz_df = pd.DataFrame({
                'PC1': pca_result[:, 0],
                'PC2': pca_result[:, 1],
                'Cluster': cluster_labels,
                'Title': [article['title'][:50] + '...' if len(article['title']) > 50 else article['title'] for article in self.articles_data],
                'Category': [article['category'] for article in self.articles_data],
                'Site': [article['site'] for article in self.articles_data],
                'URL': [article['url'] for article in self.articles_data]
            })
            
            st.success(f"✅ Successfully created {n_clusters} clusters from {len(self.articles_data)} articles")
            return viz_df, vectorizer, kmeans
            
        except Exception as e:
            st.error(f"❌ Error during clustering: {str(e)}")
            return None

def create_streamlit_dashboard():
    """Create the main Streamlit dashboard"""
    st.set_page_config(
        page_title="Zimbabwean News Clustering Platform",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 Zimbabwean News Clustering Platform")
    st.markdown("---")
    
    # Initialize session state
    if 'spider' not in st.session_state:
        st.session_state.spider = NewsSpider()
    if 'clustering_done' not in st.session_state:
        st.session_state.clustering_done = False
    if 'viz_df' not in st.session_state:
        st.session_state.viz_df = None

    # Sidebar controls
    st.sidebar.header("🕷️ Spider Controls")
    
    if st.sidebar.button("🔍 Discover URL Structures", type="primary"):
        st.session_state.spider.discover_url_structures()
        
    if st.sidebar.button("📰 Scrape Articles"):
        max_articles = st.sidebar.slider("Max articles per category", 5, 20, 10)
        st.session_state.spider.scrape_articles(max_articles)
        
    if st.sidebar.button("💾 Save to CSV"):
        st.session_state.spider.save_to_csv()
        
    if st.sidebar.button("🔬 Perform Clustering"):
        n_clusters = st.sidebar.slider("Number of clusters", 2, 8, 4)
        result = st.session_state.spider.perform_clustering(n_clusters)
        if result:
            st.session_state.viz_df, st.session_state.vectorizer, st.session_state.kmeans = result
            st.session_state.clustering_done = True

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Clustering Results")
        
        if st.session_state.clustering_done and st.session_state.viz_df is not None:
            # Cluster visualization
            fig = px.scatter(
                st.session_state.viz_df,
                x='PC1', y='PC2',
                color='Cluster',
                hover_data=['Title', 'Category', 'Site'],
                title="Article Clusters (PCA Visualization)",
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster selection
            st.subheader("🎯 Explore Clusters")
            selected_cluster = st.selectbox(
                "Select a cluster to explore:",
                options=sorted(st.session_state.viz_df['Cluster'].unique()),
                format_func=lambda x: f"Cluster {x}"
            )
            
            # Display articles in selected cluster
            cluster_articles = st.session_state.viz_df[st.session_state.viz_df['Cluster'] == selected_cluster]
            
            st.write(f"**Articles in Cluster {selected_cluster}** ({len(cluster_articles)} articles)")
            
            for idx, row in cluster_articles.iterrows():
                with st.expander(f"📄 {row['Title']} - {row['Site']}"):
                    st.write(f"**Category:** {row['Category']}")
                    st.write(f"**Site:** {row['Site']}")
                    st.write(f"**URL:** [{row['URL']}]({row['URL']})")
                    
                    # Find the full article data
                    article_data = next((a for a in st.session_state.spider.articles_data if a['url'] == row['URL']), None)
                    if article_data:
                        st.write("**Content Preview:**")
                        st.write(article_data['content'][:300] + "...")
        else:
            st.info("👆 Use the sidebar controls to start the scraping and clustering process!")
    
    with col2:
        st.header("📈 Statistics")
        
        if st.session_state.spider.articles_data:
            articles_df = pd.DataFrame(st.session_state.spider.articles_data)
            
            # Articles by category
            if 'category' in articles_df.columns:
                category_counts = articles_df['category'].value_counts()
                fig_bar = px.bar(
                    x=category_counts.index,
                    y=category_counts.values,
                    title="Articles by Category",
                    labels={'x': 'Category', 'y': 'Count'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Articles by site
            if 'site' in articles_df.columns:
                site_counts = articles_df['site'].value_counts()
                fig_pie = px.pie(
                    values=site_counts.values,
                    names=site_counts.index,
                    title="Articles by News Site"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Summary metrics
            st.metric("Total Articles", len(articles_df))
            if 'category' in articles_df.columns:
                st.metric("Categories Found", articles_df['category'].nunique())
            if 'site' in articles_df.columns:
                st.metric("News Sites", articles_df['site'].nunique())
        
        # URL Structure Discovery Results
        st.header("🔍 Discovered URL Structures")
        if st.session_state.spider.url_structures:
            for site, categories in st.session_state.spider.url_structures.items():
                with st.expander(f"📰 {urlparse(site).netloc}"):
                    if categories:
                        for category, urls in categories.items():
                            st.write(f"**{category}:** {len(urls)} URLs found")
                            for url in urls[:3]:  # Show first 3 URLs
                                st.write(f"- {url}")
                            if len(urls) > 3:
                                st.write(f"... and {len(urls) - 3} more")
                    else:
                        st.write("No category URLs discovered")

if __name__ == "__main__":
    create_streamlit_dashboard()