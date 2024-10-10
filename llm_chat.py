from services.loader import FileLoader
from services.groq_model import GroqModel
from services.embed import Embed

embed = Embed()

print("""
-----------------------------------
    Tutorial to use Assalim's LLM:
      
        Upload files:
            - If you press 1, you will need to provide one full file path from your computer.
            - If you press 2, you will redirected to LLM to make questions.
            
        LLM: 
            - If your quest is on documents context, LLM will answer you with that, else the LLM will search on internet.
            - If you type `exit`, the chat will be closed.
-----------------------------------
""")

print("============ Add files to VectorStore ============")

while True:
    option = input("Add one more file to vectorStore?\n[1] - Yes\n[2] - No\n -> ")
    
    if option != "1":
        break
    
    file_path = input("Insert file path: ")
    loader_documents = FileLoader(file_path).load_file()
    embed.save_in_vectorstore(loader_documents)


groq_model = GroqModel()

while True:
    query = input("Quest something: ")
    if query == "exit":
        break

    context = embed.search_in_vectorstore(query=query)
    context = "\n".join([res.page_content for res, score in context if score <= 0.6])
    response = groq_model.answer_quest(query=query, context=context)
    
    print(f"\nLLM Response: {response}\n")


print("Thanks to use Assalim's LLM.")