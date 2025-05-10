from langchain_ollama import OllamaLLM  
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def generate_dog_names(animal_type, color):
    llm = OllamaLLM(model="mistral", temperature=0.6) # Temperature define o grau de aleatoriedade
    prompt_animal_name = PromptTemplate(
        input_variables=["animal_type", "color"],
        template="Adotei um {animal_type} {color} e estou em dúvida sobre qual nome escolher, consegue me sugerir 5 nomes interessantes por favor?"
    )
    animal_name_chain = LLMChain(
        llm=llm,
        prompt=prompt_animal_name
    )
    response = animal_name_chain.run(animal_type=animal_type, color=color) # Prompt para o modelo
    return response

if __name__ == "__main__":
    print(generate_dog_names("elefante", "rosa"))
