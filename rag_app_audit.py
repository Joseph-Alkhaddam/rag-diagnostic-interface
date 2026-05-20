# -*- coding: utf-8 -*-
flake8: noqa: F401 # This disables the linting statements for 'imported but unused'
"""
Created on Mon May 11 14:40:08 2026

@author: fowak

RAG application audit

Audit format: Title - Definitions - Examples - Sources
"""

import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

"""
Text splitter imports:
    These are the various modules imported that are used for the text splitter functions.
    
Definition:
    Literal: 
        Bring over necessary functions, methods, classes, and/or live objects that are
        necessary to execute the code and goals.
    
    Linear:
        import os 
            -> Brings over the os module, which has various functions and methods
            that extract and use files on the system's directories.
        
        import json 
            -> Brings over the json module, which has functions and methods for 
            managing and manipulating strings, lists, and dictionaries into and out of JSON
            formats.
        
        from langchain_community.document_loaders import PyPDFLoader 
            -> Brings over a custom class that provides methods to load and parse PDF documents.
        
        from langchain_huggingface import HuggingFaceEmbeddings 
            -> Brings over a custom class that acts a sentence transformer. This is used to 
            compute query embedding on sentences.
        
        from langchain_experimental.text_splitter import SemanticChunker 
            -> Brings over a custom class that can split text on semantic similarity. At a high 
            level, this splits into sentences, then groups into groups of 3 sentences, and then
            merges one that are similar in the embedding space.
        
Examples:
    *No examples needed here*
        
Sources:
    https://docs.python.org/3.11/library/os.html
    https://docs.python.org/3.11/library/json.html
    help(PyPDFLoader)
    help(HuggingFaceEmbeddings)
    https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
"""


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

"""
Warning disable:
    This line is to disable a warning that occurs during the semantic chunking process.
    
Definition:
    Literal: 
        Disable the HuggingFace warning about SYMLINKS by tricking the library into thinking 
        we explicitly turned the warning off in our system settings.
    
    Linear:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" 
            -> `os.environ` is a dictionary-like object containing all active environment 
            variables in the computer's RAM. 
            -> By assigning the string "1" (True) to the key "HF_HUB_DISABLE_SYMLINKS_WARNING", 
            we inject this setting into the active process memory. When the HuggingFace library 
            boots up a few lines later, it checks this dictionary, sees the "1", and suppresses the 
            console warning.

Examples:
    os.environ["MY_CUSTOM_MODE"] = "DEBUG"
        -> The active Python process now holds "DEBUG" in its memory. If a downstream function checks 
        `os.getenv("MY_CUSTOM_MODE")`, it will return "DEBUG". 
        (Note: This does NOT write to a physical .env file on the disk).
        
Sources:
    help(os.environ)
"""


def semantic_chunker(file_path: str):
    print(f"[SYSTEM] Initializing PyPDFLoader for: {file_path}")
    loader = PyPDFLoader(file_path)
    raw_documents = loader.load()
    print(f"[SYSTEM] Successfully extracted {len(raw_documents)} pages from PDF.")
    
"""
PDF File loading:
    This section loads a PDF file into the script.
    
Definition:
    Literal:
        Using the file path inputted, find the PDF file, and load it into the script.
        
    Linear:
        def semantic_chunker(file_path: str):
            -> Defines the function with the required argument being a string of the
            file path of the PDF file.
            
        print(f"[SYSTEM] Initializing PyPDFLoader for: {file_path}")
            -> Debugging statement for the console stating the initiation of the file
            loading.
            
        loader = PyPDFLoader(file_path)
            -> Instantiates the variable (loader) to a custom object of the PyPDFLoader
            class.
            -> Using the file path, this object has custom methods to load and parse
            the PDF file.
            
        raw_documents = loader.load()
            -> Executes the .load() method from the PyPDFLoader object. 
            -> This does not just extract text; it converts the PDF into a list of 
            LangChain `Document` objects. 
            -> Each object in the list represents one physical page and contains both 
            the raw text (`page_content`) and the extraction data (`metadata`).
            
        print(f"[SYSTEM] Successfully extracted {len(raw_documents)} pages from PDF.")
            -> Debugging statement for the console stating the successful extraction
            and loading of the PDF file.
            
Sources:
    help(PyPDFLoader)
    help(PyPDFLoader.load)
"""


print("[SYSTEM] Booting up local HuggingFace embedding model...")
local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

"""
Free small AI:
    This section is for making the sentence transformer used in the semantic chunker.
    
Definition:
    Literal:
        Create the small AI that will be used by the document splitter for semantics.
        
    Linear:
        print("[SYSTEM] Booting up local HuggingFace embedding model...")
            -> Debugging statement for the console stating the initiation of the semantic
            chunker's embedding model.
            
        local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            -> Instantiates the `HuggingFaceEmbeddings` class using the 'all-MiniLM-L6-v2' model, 
            assigning the resulting object to the `local_embeddings` variable.
            -> This object acts as the mathematical engine for the semantic chunker. It translates 
            English sentences into high-dimensional vector arrays (lists of numbers) so the chunker 
            can calculate the mathematical distance (meaning) between sentences.
            
Sources:
    help(HuggingFaceEmbeddings)
"""    


print("[SYSTEM] Configuring Semantic Chunker...")
semantic_splitter = SemanticChunker(
    embeddings=local_embeddings,
    breakpoint_threshold_type="percentile" 
)

"""
The semantic splitter:
    This section is for setting up the semantic text splitter.
    
Definition:
    Literal:
        Instantiate the semantic splitter using the local embedding model and preferred parameters.
        
    Linear:
        print("[SYSTEM] Configuring Semantic Chunker...")
            -> Debugging statement for the console stating the semantic splitter is being created.
            
        semantic_splitter = SemanticChunker(
            -> Assigning the custom SemanticChunker object to the variable semantic_splitter.
            -> This object has capabilities of splitting documents in chunks based on semantic
            similarity. 
            -> At a high level, this object can split the document into groups of sentences as well,
            and then merge similar sentences in that group into a chunk.
        
        embeddings=local_embeddings
            -> Assigning the embeddings parameter to the previously instantiated local embedding
            model object.
            -> When a method for splitting text or documents in called, the semantics of the chunk
            splitting will be based on this mini local model's vectors.
            
        breakpoint_threshold_type="percentile"
            -> Tells the algorithm how to mathematically define a "shift in topic."
            -> By using "percentile", the chunker calculates the distances between all sentences, 
            sorts them, and only physically cuts the text when the change in meaning lands in the 
            highest percentiles (default is usually the 95th percentile, meaning only the top 5% 
            most extreme topic changes trigger a cut).
            
Example:
    sensitive_splitter = SemanticChunker(
        embeddings=local_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=80.0
    )
        -> This instantiates a chunker that cuts the text much more frequently, because it 
        triggers a split at the 80th percentile instead of the default 95th. It is much more 
        "sensitive" to minor topic changes.
        
Sources:
    help(SemanticChunker)
    https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
"""


print("[SYSTEM] Pushing raw text through the Semantic Chunker. Standby...")
smart_chunks = semantic_splitter.split_documents(raw_documents)
print(f"\n[SUCCESS] Document sliced into {len(smart_chunks)} semantic chunks.")

"""
Splitting the document:
    This section is for executing the split_documents method using the SemanticChunker object.
    
Definition:
    Literal:
        Using the semantic chunker with the local embedding model and preferred splitting settings,
        split the page content of each page that raw_documets carries by semantic similarity.
        
    Linear:
        print("[SYSTEM] Pushing raw text through the Semantic Chunker. Standby...")
            -> Debugging statement for the console stating the semantic splitter has begun to run.
            
        smart_chunks = semantic_splitter.split_documents(raw_documents)
            -> Assigning the variable smart_chunks the outputted list of LangChain document objects
            produced by the semantic splitter.
            -> These chunks include the page content of the semantic cut-offs and the metadata tied
            to them.
            -> The split_documents method is mandatory for obtaining the metadata tied to each page,
            and thus chunk. LangChain document objects carry the extra metadata, and the split_text
            method will not retrieve the metadata.
            
        print(f"\n[SUCCESS] Document sliced into {len(smart_chunks)} semantic chunks.")
            -> Debugging statement for the console stating the completion of the document split and
            the amount of chunks produced in the list.
            
Sources:
    help(SemanticChunker)
"""


print("\n--- SAMPLE CHUNK 1 ---")
print(smart_chunks[0].page_content)
print(f"Metadata (Source File): {smart_chunks[0].metadata['source']}")
print(f"Metadata (Page Number): {smart_chunks[0].metadata['page']}")

return smart_chunks

"""
Final check and results:
    This section is for diagnostic checks of the results and returning the desired chunks of text.
    
Definition:
    Literal:
        To test the results of the semantic chunks produced, display the first result, its source
        file, and its page number. Finally, return the entire list of chunks from the functions's
        scope.
        
    Linear:
        print("\n--- SAMPLE CHUNK 1 ---")
            -> Debugging statement for the console stating there is a following sample chunk for
            diagnostics.
            
        print(smart_chunks[0].page_content)
            -> Fetches the first item in the list smart_chunks
            -> Each item is a LangChain document object, which act closely as dictionaries, but 
            this custom object has the attribute page_content, and it's value is fetched via dot
            notation.
            -> The value of the page_content attribute is then printed.
            
        print(f"Metadata (Source File): {smart_chunks[0].metadata['source']}")
            -> Debugging statement for the console stating the source file value from the metadata
            attribute on the first item (item 0) in smart_chunks.
            
        print(f"Metadata (Page Number): {smart_chunks[0].metadata['page']}")
            -> Debugging statement for the console stating the page number value from the metadata 
            attribute on the first item (item 0) in smart_chunks.
            
       return smart_chunks
            -> Acts as the output valve of the function. It passes the processed list of LangChain 
            objects back to whatever script called it. 
            -> (Note: It does not save to global memory automatically; the calling script must 
            catch and assign this output to a variable to keep it in RAM).
"""


import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

"""
Vector embedding imports:
    These are the various modules imported that are used for the vector embedding functions.
    
Definition:
    Literal:
        Bring over necessary functions, methods, classes, and/or live objects that are
        necessary to execute the code and goals.
    
    Linear:
        import os 
            -> Brings over the os module, which has various functions and methods
            that extract and use files on the system's directories.
        
        from openai import OpenAI
            -> Brings over the OpenAI class, which becomes a synchronous OpenAI API client when
            instantiated.
            
        from pinecone import Pinecone
            -> Brings over the Pinecone class, which becomes a synchronous Pinecone API client when
            instantiated.
            
        from dotenv import load_dotenv
            -> Brings over the load_dotenv function, which parses a .env file and then load all 
            the variables found as environment variables.
        
Source:
    help(os)
    help(OpenAI)
    help(Pinecone)
    help(load_dotenv)
"""


























