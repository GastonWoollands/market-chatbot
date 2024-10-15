import os
import yaml
from typing import Optional
import reflex as rx
from mkt_app.reflex_chat import chat
from mkt_app.styles.styles import Size, Spacing, SizeText

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.agent.openai import OpenAIAgent

from llama_index.llms.openai import OpenAI

from pinecone.grpc import PineconeGRPC
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY   = os.getenv('OPENAI_API_KEY')

config_path = 'config.yaml'

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

tickers_dict = config['details']
tickers      = config['tickers']


#-----------------------------------------------------------------------------------

def get_pc_index(pinecone_api_key: str, ticker: str, tickers_dict: dict = tickers_dict):
    """Create pinecone connection for the ticker DB"""

    pc = PineconeGRPC(api_key=pinecone_api_key)

    index_name = str(ticker.lower())
    ticker_str = tickers_dict.get(ticker, '')

    pinecone_index = pc.Index(index_name)
    vector_store   = PineconeVectorStore(pinecone_index=pinecone_index)
    vector_index   = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    retriever      = VectorIndexRetriever(index=vector_index, similarity_top_k=4)

    query_engine  = RetrieverQueryEngine(retriever=retriever)

    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="query_engine",
            description=f"Useful to answer questions about {ticker} - {ticker_str} 10 Q reports",
        ),
    )

    llm = OpenAI(model="gpt-4o-mini", 
                temperature=0)

    agent = OpenAIAgent.from_tools([query_engine_tool], 
                                llm=llm,
                                system_prompt=f"assume you are a financial expert about {ticker} - ({ticker_str}) performance. only response questions related to the ticker {ticker} ({ticker_str}). be strictly detailed in the responses using the context provided",
                                verbose=False, 
                                temperature=0)
    
    return agent

agents = {_ticker: get_pc_index(pinecone_api_key = PINECONE_API_KEY, ticker = _ticker, tickers_dict = tickers_dict) for _ticker in tickers}


#-----------------------------------------------------------------------------------

class QA(rx.Base):
    """A question and answer pair."""
    question: str
    answer: str

#-----------------------------------------------------------------------------------


class FormSelectState(rx.State):
    selected_ticker: str = ""
    selected_bool: bool = False

    def handle_submit(self, form_data: dict):
        """Handle the form submit."""
        self.selected_ticker = form_data.get("ticker", "")
        self.selected_bool   = True
        print(f"Selected Ticker: {self.selected_ticker}")

def ticker_form_select(tickers: list):
    return rx.flex(
        rx.form.root(
            rx.flex(
                rx.select.root(
                    rx.select.trigger(),
                    rx.select.content(
                        rx.select.group(
                            rx.select.label("Select Ticker"),
                            *[rx.select.item(ticker, value=ticker) for ticker in tickers],
                        )
                    ),
                    # default_value=tickers[0] if tickers else "",
                    name="ticker",
                ),
                rx.button("Submit", type="submit"),
                width="100%",
                direction="column",
                spacing="2",
            ),
            on_submit=FormSelectState.handle_submit,
            reset_on_submit=False,
        ),
        rx.divider(size="4"),
        rx.heading("Selected Ticker"),
        rx.text(FormSelectState.selected_ticker), 
        width="100%",
        direction="column",
        spacing="2",
    )


#-----------------------------------------------------------------------------------

async def process(chat, question: str):
    """Get the response from the API."""

    ticker = FormSelectState.selected_ticker
    agent = agents.get(ticker, None)
    # agent = agents.get('MELI')

    print(f'{agent} - {ticker}' )

    if len(chat.messages) > 1:
        chat.messages = chat.messages[-2:]

    chat.messages.append(
        {"role": "system", "content": (
                "You are an expert financial assistant. "
                f"You are only qualified to respond to questions regarding {ticker} performance based on the 10 K report provided. "
                f"If a question is asked that is outside of this scope, respond with 'I am not capable of answering questions outside {ticker} performance.' "
                "IMPORTANT!: DO NOT RESPONDE QUESTION WITH NO REGARD TO FINANCIAL TOPICS")
            },
    )

    chat.messages.append(
        {"role": "user", "content": question}
    )

    chat.processing = True
    yield

    try:
        response = await agent.astream_chat(question)
        
        full_response = ""

        async for token in response.async_response_gen():
            full_response += token

            if chat.messages[-1]["role"] == "assistant":
                chat.messages[-1]["content"] = full_response
            else:
                chat.messages.append({"role": "assistant", "content": full_response})
            yield

    except Exception as e:
        chat.processing = False
        yield rx.window_alert(f"{ticker} - {agent}")
        yield rx.window_alert(f"There is an error in the server: {str(e)}. Try again later.")
        return

    if len(chat.messages) > 1:
        chat.messages = chat.messages[-2:]

    chat.processing = False
#-----------------------------------------------------------------------------------


def index() -> rx.Component:
    return rx.center(
        rx.vstack(
            ticker_form_select(tickers),
            
            rx.cond(
                FormSelectState.selected_bool,
                rx.vstack(
                    rx.heading("AI Assistant", font_size=Size.DEFAULT.value, align='center'),
                    rx.text(
                        "Ask my AI assistant anything and unlock more insights!",
                        font_size=SizeText.SMALL.value,
                        align='center',
                        white_space="normal"
                    ),
                    chat(process=process),
                    spacing=Spacing.DEFAULT.value,
                    align='center',
                    justify='between',
                    white_space="normal",
                    padding_y=Size.SMALL.value,
                    padding_right=Size.SMALL.value,
                    padding_left=Size.SMALL.value,
                    padding_x=Size.SMALL.value,
                ),

                rx.text("Please select a ticker to continue.", font_size=SizeText.SMALL.value, align='center'),
            ),

            rx.spacer(direction='column', align='center', spacing=Spacing.DEFAULT.value),

            max_width='540px',
            width="100%",
            margin_top="2em",
            margin_bottom="2em",
            padding='2em',
            border_radius="1em",
            align="center",
            background_color=rx.color("mauve", 1),
            spacing=Spacing.LARGE.value,
            justify='between',
            white_space="normal",
        ),
        width="100%",
        background=(
            "radial-gradient(circle at 22% 11%, rgba(62, 180, 137, .40), "
            "hsla(0, 0%, 100%, 0) 19%), radial-gradient(circle at 82% 25%, "
            "rgba(33, 150, 243, .36), hsla(0, 0%, 100%, 0) 35%), "
            "radial-gradient(circle at 25% 61%, rgba(250, 128, 114, .56), "
            "hsla(0, 0%, 100%, 0) 55%)"
        ),
    )

app = rx.App(
    theme=rx.theme(
        appearance="dark", has_background=True, radius="medium", accent_color="mint"
    ),
    style={
        rx.text: {"font_size": "15px"},
        rx.link: {"font_size": "15px"},
        rx.code: {"font_size": "15px"},
    },
)
app.add_page(index)