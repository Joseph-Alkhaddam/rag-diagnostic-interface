# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:53:43 2026

@author: fowak

Pinecone Manager
"""

import os
from pinecone import (Pinecone, 
                      ServerlessSpec, 
                      CloudProvider, 
                      AwsRegion, 
                      Metric, 
                      DeletionProtection, 
                      VectorType)
from dotenv import load_dotenv


def index_wipe(index_name: str):
    load_dotenv()
    pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc_client.Index(index_name)
    index.delete(
        delete_all=True,
        namespace='__default__'
        )
    
    
def create_index(index_name: str):
    load_dotenv()
    pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if not pc_client.has_index(index_name):
        pc_client.create_index(
            name=index_name,
            spec=ServerlessSpec(
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1,
                ),
            dimension=1536,
            metric=Metric.COSINE,
            vector_type=VectorType.DENSE,
            deletion_protection=DeletionProtection.ENABLED
            )
    if pc_client.has_index(index_name):
        print("Successfully created new index")
    
    
new_index_name = "2009-infinit-g37x-data"

create_index(new_index_name)
   