from DocReader.predict import Predictor
from DocRetriever.indexer import Indexer


if __name__ == "__main__":
    indexer = Indexer("DocRetriever/Demo")
    # indexer.build_index()
    # indexer.dumps()
    predictor = Predictor()
    while True:
        question = input("Please input question:\n")
        contexts = indexer.search(question)
        if len(contexts) == 0:
            print("Can't find the answer!")
        else:
            predictions = []
            for context in contexts:
                predictions.append((predictor.get_prediction(question, context), context))
            all_predictions = list(sorted(predictions, key=lambda d: d[0][1], reverse=True))[:3]
            for i, predict in enumerate(all_predictions):
                print("Rank {0}\nPredict Answer: {1}\nOriginal Sentence: {2}\nScore:{3}\n\n".format(i, predict[0][0], predict[1], predict[0][1]))