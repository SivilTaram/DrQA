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
                predictions.append(predictor.get_prediction(question, context))
            prediction = list(sorted(predictions, key=lambda d: d[1], reverse=True))[:3]
            print("The answer is: {0}".format(prediction))