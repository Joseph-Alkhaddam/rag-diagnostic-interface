# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:53:43 2026
@author: fowak

Pinecone Database Lifecycle & Management Utility
"""

import os
from typing import List
from pinecone import (
    Pinecone, 
    ServerlessSpec, 
    CloudProvider, 
    AwsRegion, 
    Metric, 
    VectorType
)
from dotenv import load_dotenv

# Initialize credentials once at module level for utility functions
load_dotenv()
PINECONE_KEY = os.getenv("PINECONE_API_KEY")

def _get_client() -> Pinecone:
    """Internal helper to safely initialize the Pinecone client."""
    if not PINECONE_KEY:
        raise ValueError("[ERROR] PINECONE_API_KEY not found in environment.")
    return Pinecone(api_key=PINECONE_KEY)


def create_pinecone_index(index_name: str, dimension: int = 1536) -> None:
    """
    Provisions a new Serverless Vector Index on Pinecone.

    Parameters
    ----------
    index_name : str
        The desired name for the new index (must be lowercase, alphanumeric, and hyphens).
    dimension : int, optional
        The mathematical dimension of the vectors. Default is 1536 (OpenAI text-embedding-3-small).
    """
    print(f"\n[SYSTEM] Attempting to provision new index: '{index_name}'...")
    pc = _get_client()
    
    try:
        existing_indexes = pc.list_indexes().names()
        if index_name in existing_indexes:
            print(f"[WARNING] Index '{index_name}' already exists. Aborting creation.")
            return

        pc.create_index(
            name=index_name,
            spec=ServerlessSpec(
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1,
            ),
            dimension=dimension,
            metric=Metric.COSINE,
            vector_type=VectorType.DENSE,
            # Note: Deletion protection removed for easier dev-cycle teardowns. 
            # Add it back for strict production environments.
        )
        print(f"[SUCCESS] Index '{index_name}' successfully provisioned.")
        
    except Exception as e:
        print(f"[ERROR] Failed to create index. Details: {str(e)}")


def wipe_index_data(index_name: str, namespace: str = "") -> None:
    """
    Deletes all vector data inside an index without destroying the index infrastructure itself.
    
    Parameters
    ----------
    index_name : str
        The target vector index identifier.
    namespace : str, optional
        The specific namespace to clear. Defaults to the empty string (default namespace).
    """
    print(f"\n[SYSTEM] Initiating data wipe for index: '{index_name}' (Namespace: '{namespace}')...")
    pc = _get_client()
    
    try:
        index = pc.Index(index_name)
        index.delete(delete_all=True, namespace=namespace)
        print(f"[SUCCESS] All vectors successfully purged from '{index_name}'.")
    except Exception as e:
        print(f"[ERROR] Failed to wipe index data. Details: {str(e)}")


def delete_pinecone_index(index_name: str) -> None:
    """
    Completely tears down and destroys a Pinecone index to free up project limits.

    Parameters
    ----------
    index_name : str
        The target vector index identifier to destroy.
    """
    print(f"\n[SYSTEM] Initiating complete teardown of index: '{index_name}'...")
    pc = _get_client()
    
    try:
        existing_indexes = pc.list_indexes().names()
        if index_name not in existing_indexes:
            print(f"[WARNING] Index '{index_name}' does not exist. Nothing to delete.")
            return
            
        pc.delete_index(index_name)
        print(f"[SUCCESS] Index '{index_name}' has been permanently destroyed.")
    except Exception as e:
        print(f"[ERROR] Failed to delete index. Details: {str(e)}")


def list_active_indexes() -> List[str]:
    """
    Retrieves and prints all currently active indexes on the Pinecone project.
    
    Returns
    -------
    List[str]
        A list of active index names.
    """
    print("\n[SYSTEM] Fetching active cloud indexes...")
    pc = _get_client()
    
    try:
        indexes = pc.list_indexes().names()
        if not indexes:
            print(" -> No active indexes found on this project.")
            return []
            
        print(f" -> Active Indexes ({len(indexes)} found):")
        for idx in indexes:
            print(f"    - {idx}")
        return indexes
    except Exception as e:
        print(f"[ERROR] Failed to fetch indexes. Details: {str(e)}")
        return []


def print_index_stats(index_name: str) -> None:
    """
    Fetches the total vector count and metric status for a specific index.
    Crucial for verifying if an ingestion pipeline succeeded.

    Parameters
    ----------
    index_name : str
        The target vector index identifier.
    """
    print(f"\n[SYSTEM] Fetching telemetry for index: '{index_name}'...")
    pc = _get_client()
    
    try:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        
        print("--- INDEX DIAGNOSTICS ---")
        print(f"Dimension : {stats.dimension}")
        print(f"Total Vectors: {stats.total_vector_count}")
        print("Namespaces:")
        for ns, info in stats.namespaces.items():
            ns_name = ns if ns else "Default"
            print(f"  - {ns_name}: {info.vector_count} vectors")
        print("-------------------------")
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch index stats. Details: {str(e)}")


# --- ENTRY POINT FOR STANDALONE EXECUTION ---
if __name__ == "__main__":
    # ---------------------------------------------------------
    # DevOps Control Panel
    # Uncomment the function you wish to run and execute the file.
    # ---------------------------------------------------------
    
    TARGET_INDEX = ""
    
    # 1. Inspect your infrastructure
    # list_active_indexes()
    # print_index_stats(TARGET_INDEX)
    
    # 2. Provision new infrastructure
    create_pinecone_index(TARGET_INDEX)
    
    # 3. Clean or destroy infrastructure
    # wipe_index_data(TARGET_INDEX)
    # delete_pinecone_index(TARGET_INDEX)
    
    print("\n[READY] Comment out the desired function in the execution block to run.")
    
    
