javac *.java
echo ""
echo "-------------------------------------------"
echo "-----------  SIMPLE PERCEPTRON ------------"
echo "-------------------------------------------"
java simplePerceptron CVSplits phishing.train phishing.dev phishing.test

echo ""
echo "-------------------------------------------"
echo "-----------  DYNAMIC PERCEPTRON -----------"
echo "-------------------------------------------"
java dynamicPerceptron CVSplits phishing.train phishing.dev phishing.test

echo ""
echo "-------------------------------------------"
echo "------------  MARGIN PERCEPTRON -----------"
echo "-------------------------------------------"
java marginPerceptron CVSplits phishing.train phishing.dev phishing.test

echo ""
echo "-------------------------------------------"
echo "----------  AVERAGED PERCEPTRON -----------"
echo "-------------------------------------------"
java averagedPerceptron CVSplits phishing.train phishing.dev phishing.test

echo ""
echo "-------------------------------------------"
echo "-----------  MAJORITY BASELINE  -----------"
echo "-------------------------------------------"
java majorityBaseline CVSplits phishing.train phishing.dev phishing.test
