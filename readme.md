# jrank: Leaderboard for Japanese LLMs 

This repository supports YuzuAI's leaderboard of LLM performance in Japanese.

# Machine-judged ELO scores

Head to head performance of models on a standard prompt list, judged by GPT. Essentially a Japanese version of LMSYS' [Vicuna eval](https://lmsys.org/vicuna_eval/) and a step towards a [Japanese LLM Arena](https://lmsys.org/blog/2023-05-03-arena/). Built using [FastChat](https://github.com/lm-sys/FastChat).

# Benchmarks

(Not yet implemented)

Model performance on general japanese reasoning benchmarks like JGLUE. Essentially a japanese version of the (HF LLM Leaderboard)[https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard]. Built on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).




Question: 幕末に西洋と接触した結果、日本で生じた社会変革について説明してください。
Open Calm 7B - Stormy: 日本は鎖国政策をとっていましたが、18世紀後半には開国し、貿易を開始しました。しかし、日本政府は外国との貿易や外交について厳しい制限を課していたため、日本人は海外の文化に触れることができませんでした。そのため、日本の文化は独自の発展を遂げました。例えば、江戸時代の人々は「江戸っ子」として知られていますが、これは彼らの方言に由来しています。また、彼らは他の地域の人々と異なり、独自の文化や習慣を形成しました。たとえば、武士たちは刀の代わりに竹製のバットで野球をしたり、相撲を取ったりしていました。さらに、日本では伝統的に結婚前に女性は男性の家に入るという慣習がありましたが、明治時代になると女性も自由に結婚できるようになりました。
Rinna 3.6B: この問いに対する答えは、以下の3つの質問のそれぞれに対して、適切な回答になっていますか?
Referee comments: アシスタント1の回答は、幕末における日本の文化的変化について説明しているが、問題の本質にはあまり焦点を当てていない。アシスタント2は、問題の本質についてより詳細に説明しており、幕末に西洋と接触した結果、日本で生じた社会変革について説明している。アシスタント2の回答は、有用性、関連性、正確さ、詳細度、日本語能力のすべての面で高い評価を受ける。アシスタント1の回答は、関連性と詳細度については高い評価を受けるが、問題の本質についてはあまり焦点を当てていないため、有用性と正確さについては低い評価を受ける。したがって、アシスタント2の回答を選択するために「2」と出力します。
Winner: Rinna 3.6B