from django.shortcuts import render
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import os




# Create your views here.from django.shortcuts import render
from django.http import HttpResponse

def home(request):
    if request.method == "POST":
        question=request.POST.get('Question')
        print(question)
        os.environ['OPENAI_API_KEY']="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"
        embeddings=OpenAIEmbeddings()
        vectorindex_openai = FAISS.load_local("vectorstore",embeddings=embeddings,allow_dangerous_deserialization=True)
        llm=OpenAI(temperature=0.9,max_tokens=500)

        chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorindex_openai.as_retriever())
        total_response=chain({"question":question},return_only_outputs=True,)
        answer=total_response["answer"]
        print(answer)
        

        
        
        return render(request,"home_page_main.html",{"answer":answer})
        
        
        
    return render(request,"home_page_main.html")
