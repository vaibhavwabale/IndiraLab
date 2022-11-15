1. ChatBot 

dict = {
    "hi": "Hello",
    "Who are you ?": "I am ChatBot",
    "What is your name ?": "My name is Vaibhav",
    "What is the name of your institute ?": "ICCS",
    "Percentage in Graduation ?": "76%"
}

a = input()

if a in dict:
    print(dict[a])
else:
    print("Invalid Input, I don't know the answer !!")
 
======================================================================

2. Area of Square
def area():
    a = int(input("Enter Side"))  
    area = a * a 
    print(area)

======================================================================

3. Area of Triangle
def triangle():
    base = int(input("Enter Base"))
    height = int(input("Enter Height"))
    areaTri = 0.5 * base * height
    print(areaTri)
    
=====================================================================

4. Area of Circle
def circle():
    r = int(input("Enter Radius"))
    circle = 3.14 * r * r
    print(circle)
circle()

======================================================================
