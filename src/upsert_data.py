import os
import yaml
import argparse
from pathlib import Path
from llmsherpa.readers import LayoutPDFReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import Document

from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC
from llama_index.vector_stores.pinecone import PineconeVectorStore

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

config_path = 'config.yaml'

#---------------------------------------------------------

def create_index(pc_object, idx_name: str, dimension: int = 1536, metric: str = "cosine"):
    """Create pinecone index"""

    pc_object.create_index(
        name=idx_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    )

#---------------------------------------------------------

def validate_index(idx_name: str, pinecone_api: str):
    """Validate index existance"""

    _pc = PineconeGRPC(api_key=pinecone_api)

    idx_names = [idx['name'] for idx in _pc.list_indexes()]

    if idx_name not in idx_names:
        create_index(
            pc_object = _pc,
            idx_name = idx_name
            )
        print(f'Index: {idx_name} - created succesfully')

    else:
        print(f'Index: {idx_name} - already exists')
        pass

#---------------------------------------------------------

def main(pinecone_api_key: str, openai_api_key: str):

    pc = PineconeGRPC(api_key=pinecone_api_key)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    base_directory = Path(config['base_directory'])
    tickers        = config['tickers']
    years          = config['years']
    llmsherpa_url  = config['llmsherpa_url'] # Use Docker image
    tickers_dict   = config['details']

    Settings.embed_model = OpenAIEmbedding(api_key=openai_api_key)

    pdf_reader = LayoutPDFReader(llmsherpa_url)

    for ticker in tickers:
        for year in years:
            file_path = str(base_directory / year / f'{ticker}-{year}-10K.pdf')

            ticker_idx = str(ticker.lower())
            ticker_str = tickers_dict.get(ticker, '')

            validate_index(idx_name = ticker_idx, pinecone_api=pinecone_api_key)

            pinecone_index  = pc.Index(ticker_idx)
            vector_store    = PineconeVectorStore(pinecone_index=pinecone_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            index = VectorStoreIndex([], show_progress=False, storage_context=storage_context)
            
            doc = pdf_reader.read_pdf(file_path)

            for chunk in doc.chunks():
                index.insert(Document(text=chunk.to_context_text(), extra_info = {
                    'description': f'Document about {ticker} ({ticker_str}) 10-K {year} SEC report'}))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Upload documents to Pinecone. Provide Pinecone API Key - OpenAI API Key")
    
    parser.add_argument('--pinecone_api_key', '-p'  , type=str, required=True, help="Pinecone API Key")
    parser.add_argument('--openai_api_key'  , '-oai', type=str, required=True, help="OpenAI API Key"  )
    args = parser.parse_args()

    main(pinecone_api_key = args.pinecone_api_key, 
         openai_api_key   = args.openai_api_key )


