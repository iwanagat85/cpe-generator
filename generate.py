import torch
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
embedding_model_name = "intfloat/multilingual-e5-base"
lora_path = "outputs/20250212171602"


def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    peft_model = PeftModel.from_pretrained(model, get_last_checkpoint(lora_path)).to(device)
    return peft_model, tokenizer


def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=embedding_model_name)


def load_vector_store(embeddings):
    return Chroma(persist_directory="vectorstore", embedding_function=embeddings)


def get_output_parser():
    response_schemas = [
        ResponseSchema(name="part", description="May have 1 of 3 values. a for Applications. h for Hardware. o for Operating Systems."),
        ResponseSchema(name="vendor", description="Values for this attribute SHOULD describe or identify the person or organization that manufactured or created the product."),
        ResponseSchema(name="product", description="The name of the system/package/component. product and vendor are sometimes identical."),
        ResponseSchema(name="version", description="The version of the system/package/component."),
    ]
    return StructuredOutputParser.from_response_schemas(response_schemas)


def get_prompt_template(output_parser):
    format_instructions = output_parser.get_format_instructions()
    return PromptTemplate(
        template=(
            "Generate a JSON from the given text.\n"
            "{format_instructions}\n\n"
            "Please refer to the information below.\n\n"
            "### Following information:\n"
            "{context}\n\n"
        ),
        input_variables=["context"],
        partial_variables={"format_instructions": format_instructions}
    )


@torch.no_grad()
def generate(query):
    model, tokenizer = load_model_and_tokenizer()

    embeddings = load_embedding_model()
    vectorstore = load_vector_store(embeddings)

    output_parser = get_output_parser()
    prompt_template = get_prompt_template(output_parser)

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.85,
        },
    )
    retrieved_docs = retriever.invoke(query)

    # context_text = "\n".join([doc.page_content for doc in retrieved_docs])
    chat_template = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": prompt_template.format(context=retrieved_docs)},
            {"role": "user", "content": query}
        ],
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([chat_template], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128,
        num_return_sequences= 10,
        do_sample=True,
        temperature=1.3,
    )

    generated_ids = [
        output_ids[len(model_inputs.input_ids[0]):] for output_ids in generated_ids
    ]

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    print(f"Input: {query}")
    print("Output:")
    for idx, response in enumerate(responses):
        try:
            result = output_parser.parse(response)
            part = result["part"]
            vendor = result["vendor"]
            product = result["product"]
            version = result["version"]
            print(f"#{idx}: cpe:2.3:{part}:{vendor}:{product}:{'*' if version=='' or version=='ANY' else version}:*:*:*:*:*:*:*")
        except Exception as e:
            print(f"#{idx}: {response}")
            print(e)


def main():
    queries = [
        "Visual Studio Code 0.2.9",
        "Hitachi Community Plugin Framework 6.1.0.13-r",
    ]
    for query in queries:
        generate(query)


if __name__ == "__main__":
    main()
