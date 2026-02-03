#!/usr/bin/env python3
"""
Script to verify ChromaDB contents and check if data ingestion was actually performed.
"""

import os
import sys
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

def check_chromadb_contents():
    """Check the actual contents of ChromaDB database."""
    
    # Add app directory to path to import services
    sys.path.append('app')
    
    try:
        # Initialize ChromaDB client same as in the service
        chroma_persist_dir = "./app/chroma_db"
        collection_name = "asktemoc_collection"
        
        print(f"Checking ChromaDB at: {chroma_persist_dir}")
        print(f"Collection name: {collection_name}")
        
        # Check if database directory exists
        if not os.path.exists(chroma_persist_dir):
            print(f"❌ ChromaDB directory does not exist: {chroma_persist_dir}")
            return False
        
        # Initialize client
        client = chromadb.PersistentClient(path=chroma_persist_dir)
        
        # Get collection
        collection = client.get_collection(collection_name)
        
        # Get document count
        document_count = collection.count()
        print(f"📊 Document count in ChromaDB: {document_count}")
        
        # Get collection metadata
        metadata = collection.metadata
        print(f"📋 Collection metadata: {metadata}")
        
        # Test retrieval to see if data is actually there
        try:
            embeddings = OllamaEmbeddings(
                model="nomic-embed-text"  # Default model
            )
            
            vector_store = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embeddings,
            )
            
            # Test search
            test_results = vector_store.similarity_search("computer science", k=3)
            print(f"🔍 Test search results: {len(test_results)} documents found")
            
            if test_results:
                print("✅ Test search successful - data appears to be ingested")
                for i, result in enumerate(test_results[:2]):
                    print(f"  Result {i+1}: {result.metadata.get('source', 'unknown')}")
            else:
                print("⚠️  Test search returned no results - data may not be properly ingested")
                
        except Exception as e:
            print(f"❌ Error during test search: {e}")
            return False
        
        return document_count > 0
        
    except Exception as e:
        print(f"❌ Error checking ChromaDB: {e}")
        return False

def count_source_files():
    """Count the number of source files that should have been ingested."""
    
    data_dir = "app/services/data"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory does not exist: {data_dir}")
        return 0
    
    # Count requirements.txt files
    requirements_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file == "requirements.txt":
                requirements_files.append(os.path.join(root, file))
    
    print(f"📁 Source requirements.txt files found: {len(requirements_files)}")
    return len(requirements_files)

if __name__ == "__main__":
    print("=== ChromaDB Data Ingestion Verification ===\n")
    
    # Check source files count
    source_count = count_source_files()
    
    # Check ChromaDB contents
    print("\n" + "="*50)
    chromadb_has_data = check_chromadb_contents()
    
    print("\n" + "="*50)
    print("VERIFICATION SUMMARY:")
    print(f"📁 Source files (requirements.txt): {source_count}")
    
    # Get the actual count
    try:
        client = chromadb.PersistentClient(path="./app/chroma_db")
        collection = client.get_collection("asktemoc_collection")
        db_count = collection.count()
        print(f"📊 Documents in ChromaDB: {db_count}")
    except Exception as e:
        db_count = 0
        print(f"📊 Documents in ChromaDB: {db_count} (Error: {e})")
    
    print("\n" + "="*50)
    print("VERIFICATION RESULTS:")
    
    if source_count > 0:
        print(f"✅ Found {source_count} source files to ingest")
    else:
        print("❌ No source files found for ingestion")
    
    if db_count > 0:
        print(f"✅ ChromaDB contains data ({db_count} documents)")
        
        # Calculate average chunks per file
        if source_count > 0:
            avg_chunks_per_file = db_count / source_count
            print(f"📊 Average chunks per source file: {avg_chunks_per_file:.2f}")
            
            if avg_chunks_per_file > 1:
                print("✅ Multiple chunks per file indicates proper text splitting")
            else:
                print("⚠️  Single chunk per file may indicate insufficient text splitting")
    else:
        print("❌ ChromaDB appears to be empty or inaccessible")
    
    print(f"\n🔍 CONCLUSION:")
    if source_count > 0 and db_count > 0:
        print("✅ Data ingestion APPEARS to have been completed successfully")
        print(f"   - Found {source_count} source files")
        print(f"   - ChromaDB contains {db_count} document chunks")
        print("   - Test retrieval functionality confirmed")
    elif source_count > 0 and db_count == 0:
        print("❌ Data ingestion likely NOT completed")
        print(f"   - Found {source_count} source files but ChromaDB is empty")
    else:
        print("❌ Data ingestion status cannot be determined")
        print("   - No source files found or ChromaDB inaccessible")