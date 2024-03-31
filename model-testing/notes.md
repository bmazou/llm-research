## Modely
### Porovnání 7B-13B modelů
- [model-comparison](https://github.com/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/README.md)
#### Relevantní otázky
- Zajímá mě hlavně extrakce informací z textu a analýza sentimentu 
- 2,4,5,6,10,12,13,15
- Nejdůležitější otázky
	- 12 a 13: context retrieval
 
#### Zajímavý modely
- [OpenOrca-7B](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca)
- [Nous-Capybara-7B](https://huggingface.co/TheBloke/Nous-Capybara-7B-GGUF)
- Obecně se mi zdá, že 7B modely jsou zbytečně silný, pokud s nima budu používat RAG

### Menší modely
- [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)
	- [77M](https://huggingface.co/google/flan-t5-small), [248M](https://huggingface.co/google/flan-t5-base), [783M](https://huggingface.co/google/flan-t5-large), [3B](https://huggingface.co/google/flan-t5-xl)
- [Orca-mini-3B](https://huggingface.co/pankajmathur/orca_mini_3b)

#### Výsledky menších modelů
- [výsledky](./results/)
- Flan-t5 testovaný [lokálně](./test_models.py), Orca-mini přes [google-colab](./orca_mini_colab.ipynb)
- <= 783M nic moc.
- Oba 3B modely dobrý, Orca-mini je z nich výrazně lepší

## Retrieval-Augmented Generation (RAG)

### Obecně 
- [video](https://www.youtube.com/watch?v=T-D1OfcDW1M)
- Problémy LLMs:
	- Nemaj zdroj
	- Out of date
- Modelu můžu říct, ať se před odpovědí nejdřív podívá na zdroje co mu dám
- Do těch zdrojů můžu postupně přidávat nový informace, takže model nikdy nebude out of date

### Podrobněji
- [článek](https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/) mluvící víc o specifikách implementace. Porovnává RAG a finetuning
- Vytvořim si vektorovou databázi
	- Agregace a vyčištění dat
	- Chunking
		- Rozdělim data podle vět? Odstavců? Kapitol?
		- Záleží na use caseu - pro RS asi prostě pro každou Knihu/Film jeden chunk
	- Chunky embeddnu a uložim do databáze
		- Existujou na to jednoduchý tooly 
- Model dostane následují informace
	- System config
		- Popíše se modelu, co vůbec dělá a co se od něj očekává
	- History
		- Historie konverzace
	- Context
		- To, co retrievnu z databáze
	- User prompt
		- Input
- User dá prompt, retrievnu relavantní kontext z databáze, a LLM může odpovědět podle kontextu a historie
	- Dávat si pozor na token limit
- Způsoby, jak zlepšit výsledky
	- Lepší kvalita dat
	- Jinej *chunking*
	- Jinej *embedding model*
	- Změnit *system config* 

#### RAG vs fine-tuning
- Fine-tuded model se specifikuje na tu jednu kontrétní činost, takže na ostatní věci se o dost zhorší
	- Pro recommender system (RS) by to asi nevadilo
- RAG model se může jednoduše updatovat s novýma datama
	- FT by se musel pokaždý tunovat znovu
- Daj se hezky zkombinovat:
	- RS by to asi nejlíp využil tak, že se natunuje tak, aby dobře konverzoval s uživatelem jako RS system 
	- A pro RAG by to byla ta databáze knih/flmů

### Implementace RAGu
- [jupyter](https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/rag-chatbot.ipynb), [youtube](https://youtu.be/LhnCsygAvzY?si=CI7S8CmKv90zmZ0x)

## Otázky, který by bylo zajímavý zodpovědět
- Intuitivně bych řekl, že RAG model by měl bejt menši než FT model, protože hodně parametrů potřebných na samoutnou znalost knih "nahradim" tou databází
	- Je to pravda?
	- Nevyneguje se tohle zmenšení samotnou velikostí tý databáze a tím, že v ní musim hledat?
- Jak dobrá by byla grafová databáze (s nějakou heuristikou) v porovnání s vektorovou?
- Jak málo parametrů modely můžou být, aby byly stále funkční?
- U toho recommender systému bude relativně velkej problém *token limit*
	- Pokud bude malej, tak model bude zapomínat, co doporučil předtim, a pojede v kruzích
	- Řešení by mohly být buďto ten limit natvrdo zvětšit, anebo do historie nedávat celý odpovědi, ale pouze zkráceně ty knihy, co model doporučil
		- Nebo obojí
	- Možná by šlo do historie konverzace appendovat shrnutí jednodlivých otázek a odpovědí

