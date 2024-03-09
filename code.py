import os, time
import pandas as pd
from pymarc import MARCReader
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage

llm = AzureChatOpenAI(
    openai_api_key  =   os.environ["API_KEY"],
    deployment_name =   os.environ["DEPLOYMENT_NAME"],    
    azure_endpoint  =   os.environ["ENDPOINT_NAME"],
    api_version     =   os.environ["API_VERSION"],
    model_name      =   "gpt-35-turbo",
    temperature     =   0.0,
)

teach_prompt = """
As a library cataloguer, please provide the appropriate Library of Congress (LC) subject heading(s) that best match the abstract and title of the given work. The detail of the output is as follows:
- Format the subject headings in MARC 21 (fields 650, 600, 610, 651, 630) without using field 880.
- Assign subject headings that best summarise the overall contents of the work, and as specific as possible on the topics being covered.
- Include multiple subject headings if the book covers various topics. 
- Limit the number of subject headings between one to four. 
- Provide the MARC fields only, without explanations or notes or greetings.

--- Start of Examples —

Example 1:
Title: Composite property rights and boundary-treading resistance : a case study of C county in Eastern Sichuan
Summary: This thesis studies land expropriation disputes from the angle of property right, exploring its origins from the relationships between township (town) government and villagers, village collective and villagers, and different villagers, focusing on peasants' resisting low land expropriation.

650  $aLand tenure $zChina.
650  $aRight of property $zChina.

Example 2:
Title: Determinants of expertise of Olympic style Taekwondo performance
Summary: The purpose of this study was to identify the determinants of expertise and the contributory effect of domains to the Olympic style Taekwondo performance. Eighty-seven Taekwondo athletes with different levels of expertise, namely elite, sub-elite and practitioner were recruited. 

650  $aAthletes $xPsychological aspects.
650  $aAthletes $xResearch.
650  $aTae kwon do.

Example 3:
Title: An ADMM approach to the numerical solution of state constrained optimal control problems for systems modeled by linear parabolic equations
Summary: We address in this thesis the numerical solution of state constrained optimal control problems for systems modeled by linear parabolic equations. For the unconstrained or control-constrained optimal control problem, the first order optimality condition can be obtained in a general way.

650  $aFinite differences.
650  $aMultipliers (Mathematical analysis)
650  $aNumerical analysis.

--- End of Examples ---
"""

i=0
marc_6xx=""
df = pd.DataFrame()

## Loop through all MARC records
with open('2024ETD.mrc', 'rb') as fh:

    reader = MARCReader(fh,to_unicode=True, force_utf8=True)

    for record in reader:

      print("\n\n==== [ETD #"+str(i+1)+ "] ====")        
                  
      # Obtain the summary 
      marc520 = ""        
      if(len(record.get_fields('520'))>0):
        marc520 = str(record.get_fields('520')[0])
        marc520 = marc520.replace('=520','').replace('3\$a"','').replace('8\$a','').replace('3\$a','').replace('\\$a"','').replace('\\$a','')          
        marc520 = marc520.strip()
        if marc520[0]=='\\':
          marc520 = marc520[1:]
          
      # Obtain the existing subject fileds (cataloged by UC)
      marc_6xx_UC = ""
      for s in record.subjects:
        if "=655" not in str(s):
          marc_6xx_UC = marc_6xx_UC + str(s) + "\n"
      marc_6xx_UC = marc_6xx_UC.strip()
          

      # Build LLM query        
      query = teach_prompt + "\n\n Title: "+ record.title + "\n Summary: " + marc520

      # Invoke LLM
      msg = HumanMessage(content=query)
      out_ = llm.invoke([msg])
      marc_6xx = str(out_.to_json()['kwargs']['content'])        
                  
      print(query)
      print(marc_6xx)
                      
      # Save LLM results to dataframe
      data_ = {'id': str(i+1), 'Title':record.title, 'Summary':marc520, 'marc_6xx_UC': marc_6xx_UC, 'marc_6xx_ChatGPT': marc_6xx}
      if i == 0:
        df = pd.DataFrame(data_, index=[i])
      else:
        df = pd.concat([df, pd.DataFrame(data_, index=[i])], ignore_index=True)
      
      # Add 5 seconds timer in between each LLM query, to avoid hitting query limit
      time.sleep(5)

      i=i+1

df.to_csv('result.csv', index=False)
