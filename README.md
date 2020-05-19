# Latent Semantic Analysis - LA coursework project
The coursework project for Linear Algebra course (2020)

Made by Sophia Tatosh and Anna Botsula.


### GENERAL TOPIC: 

The project of ultimate encoding production from text into vectors to perform fast comparison and search of similar products in databases.

# To run the code

1) Make sure that all needed libraries are installed.
2) Run the `main.py`
3) Have fun with the model!

***

### PROBLEM

The problem we are working on - case of huge shops with enormously big databases of products.

1. Each product can be provided from different suppliers - that means that the same product can have a bunch of various descriptions. 

2. Each provider can sell his products to the number of markets and have diverse descriptions for the same product, so to change some properties of it (like cost, color, volume, etc.) sales managers have to search for special keys-identificators manually through tables or remember them somewhere (in additional table) and then synchronise.

So these two harmless points hide inside hours of useless human work for companies (money equivalent), little “human errors” can create snowballs of misunderstandings and large time-money losses. And finally - human resources can be used in more valuable ways.

### DECISION

What do we offer to solve this problem?

Create automated check to find the same products with their description using methods of Natural Language Processing based on Linear Algebra.
