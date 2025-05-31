from langchain_ollama import OllamaLLM  
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def number_of_oscars(filme, ano):
    llm = OllamaLLM(model="mistral", temperature=0.6) # Temperature define o grau de aleatoriedade
    prompt_template = PromptTemplate(
        input_variables=["filme", "ano"],
        template="Quantos oscars o filme {filme} ganhou no ano de {ano}?"
        )
    oscar_chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )
    response = oscar_chain.invoke({"filme": filme, "ano": ano}) # Prompt para o modelo
    return response

if __name__ == "__main__":
    print(number_of_oscars("Oppenheimer", "2024"))
    