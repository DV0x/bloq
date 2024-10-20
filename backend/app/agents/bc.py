import os
import json
import time
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.tools import FunctionTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
import requests
from typing import List, Dict, Any
from pinecone import Pinecone
import logging

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecentPoolAgent:
    def __init__(self, subgraph_url: str, pinecone_api_key: str, pinecone_index_name: str):
        self.subgraph_url = subgraph_url
        self.api_key = os.getenv('SUBGRAPH_API_KEY')
        if not self.api_key:
            raise ValueError("SUBGRAPH_API_KEY not found in environment variables")
        self.llm = OpenAI(temperature=0, model="gpt-4")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        self.pinecone_index = pc.Index(pinecone_index_name)
        self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        
        self.agent = self._create_agent()
        self.pools = []  # Initialize the pools attribute
        self.index = None

    def _create_agent(self):
        tools = [
            FunctionTool.from_defaults(
                fn=self.fetch_and_index_data,
                name="fetch_and_index_data",
                description="Fetch Uniswap pool data and index it in the vector store"
            ),
            FunctionTool.from_defaults(
                fn=self.answer_question,
                name="answer_question",
                description="Answer questions about the indexed pool data"
            )
        ]
        return OpenAIAgent.from_tools(tools, llm=self.llm, verbose=True)

    def graphql_request(self, user_input: str, max_retries=3, initial_delay=1) -> Dict[str, Any]:
        query = """
        query GetUniswapPools($first: Int!, $orderBy: String, $orderDirection: String) {
            pools(first: $first, orderDirection: $orderDirection) {
                id
                token0 { symbol }
                token1 { symbol }
                volumeUSD
            }
        }
        """
        variables = {
            "first": 1000,
            "orderBy": "volumeUSD",
            "orderDirection": "desc"
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.subgraph_url, 
                    json={'query': query, 'variables': variables},
                    headers=headers,
                    timeout=10  # Add a timeout
                )
                response.raise_for_status()  # Raise an exception for bad status codes
                return response.json()
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(initial_delay * (2 ** attempt))  # Exponential backoff
        
        raise Exception("Max retries reached")

    def fetch_and_index_data(self, user_input: str) -> str:
        try:
            data = self.graphql_request(user_input)
            if 'errors' in data:
                return f"Error fetching data: {data['errors']}"
            self.pools = data.get('data', {}).get('pools', [])
            if not self.pools:
                return "No pools data received from the GraphQL query"
            
            # Return the top 5 recent pools
            recent_pools = self.pools[:5]
            result = "Here are the 5 most recent pools launched on Uniswap:\n\n"
            for i, pool in enumerate(recent_pools, 1):
                result += f"{i}. {pool['token0']['symbol']}/{pool['token1']['symbol']} - Volume: ${float(pool['volumeUSD']):.2f}\n"
            
            return result
        except Exception as e:
            return f"Error during fetch and index: {str(e)}"

    def answer_question(self, question: str) -> str:
        if not self.index:
            return "No data available. Please fetch and index data first."
        try:
            query_engine = self.index.as_query_engine()
            response = query_engine.query(question)
            return str(response)
        except Exception as e:
            return f"Error answering question: {str(e)}"

    def run(self, user_input: str):
        logger.info(f"Received user input: {user_input}")
        try:
            response = self.agent.chat(user_input)
            logger.info(f"Agent response: {response.response}")
            return response.response
        except Exception as e:
            logger.error(f"Error in agent: {str(e)}")
            return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    subgraph_url = "https://gateway.thegraph.com/api/d4cf9853bdd372d0392a6628ca21921c/subgraphs/id/HUZDsRpEVP2AvzDCyzDHtdc64dyDxx8FQjzsmqSg4H3B"
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')

    agent = RecentPoolAgent(subgraph_url, pinecone_api_key, pinecone_index_name)

    
