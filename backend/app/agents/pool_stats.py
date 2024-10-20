import os
import json
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.tools import FunctionTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
import requests
from typing import List, Dict, Any
from pinecone import Pinecone

load_dotenv()

class PoolStatsAgent:
    def __init__(self, subgraph_url: str, pinecone_api_key: str, pinecone_index_name: str):
        self.subgraph_url = subgraph_url
        self.api_key = os.getenv('SUBGRAPH_API_KEY')
        if not self.api_key:
            raise ValueError("SUBGRAPH_API_KEY not found in environment variables")
        self.llm = OpenAI(temperature=0, model="gpt-4")
        
        pc = Pinecone(api_key=pinecone_api_key)
        self.pinecone_index = pc.Index(pinecone_index_name)
        self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        
        self.agent = self._create_agent()

    def _create_agent(self):
        tools = [
            FunctionTool.from_defaults(fn=self.query_pool_stats, name="query_pool_stats"),
            FunctionTool.from_defaults(fn=self.query_top_pools, name="query_top_pools")
        ]
        return OpenAIAgent.from_tools(tools, llm=self.llm, verbose=True)

    def graphql_request(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        response = requests.post(
            self.subgraph_url, 
            json={'query': query, 'variables': variables},
            headers=headers
        )
        return response.json()

    def query_pool_stats(self, pool_id: str) -> str:
        query = """
        query GetPoolStats($poolId: ID!) {
            pool(id: $poolId) {
                id
                token0 { id symbol name }
                token1 { id symbol name }
                feeTier
                liquidity
                volumeUSD
                txCount
                totalValueLockedToken0
                totalValueLockedToken1
                totalValueLockedUSD
                feesUSD
                createdAtBlockNumber
                createdAtTimestamp
            }
        }
        """
        variables = {"poolId": pool_id}
        data = self.graphql_request(query, variables)
        pool = data.get('data', {}).get('pool')
        if pool:
            return json.dumps(pool, indent=2)
        return f"No pool found with ID: {pool_id}"

    def query_top_pools(self, limit: int = 5) -> str:
        query = """
        query GetTopPools($limit: Int!) {
            pools(first: $limit, orderDirection: desc) {
                id
                token0 { symbol }
                token1 { symbol }
                volumeUSD
                txCount
                totalValueLockedUSD
            }
        }
        """
        variables = {"limit": limit}
        data = self.graphql_request(query, variables)
        pools = data.get('data', {}).get('pools', [])
        return json.dumps(pools, indent=2)

    def run(self, user_input: str):
        if "top pools" in user_input.lower():
            return self.query_top_pools(5)  # Default to top 5 pools
        elif "pool stats" in user_input.lower():
            # Extract pool ID if provided, otherwise return an error message
            pool_id = user_input.split("pool id")[-1].strip() if "pool id" in user_input else None
            if pool_id:
                return self.query_pool_stats(pool_id)
            else:
                return "Please provide a pool ID to get pool stats."
        else:
            return "I'm not sure what you're asking. You can ask for 'top pools' or 'pool stats for pool id <ID>'."

if __name__ == "__main__":
    subgraph_url = "https://gateway.thegraph.com/api/d4cf9853bdd372d0392a6628ca21921c/subgraphs/id/HUZDsRpEVP2AvzDCyzDHtdc64dyDxx8FQjzsmqSg4H3B"
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')

    agent = PoolStatsAgent(subgraph_url, pinecone_api_key, pinecone_index_name)

    user_input = "Print the top 5 pools by volume"
    print(f"User input: {user_input}")
    result = agent.run(user_input)
    print("Result:")
    print(result)
