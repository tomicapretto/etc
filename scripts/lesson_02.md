# Lesson 2: Bayesian Regression Refresher

## Concepts Introduced

* Linear Regression
    * Intercept only model
    * Models with one slope
    * Models with multiple slopes
    * Models with categorical predictors
    * Interpretation of results
    * ?Error
* PyMC
    * Model specification
    * Sampling
* Bayesian Workflow
    * Exploratory data analysis
    * Model specification
    * Model construction
    * Prior predictive sampling
    * Model fitting
    * Posterior predictive sampling
    * Model diagnostics
    * Interpretation of results

### Goal

* Get people into the code and modeling
* _Secretly_ show people formulae syntax without them knowing it
* Introduce the foundational concepts in modeling and bayesian estimation

## Section 10: Introduction

* Welcome to first lesson Regression Refresher
* In this lesson we'll take on a role familiar to those who took the intro lesson
    * Youre a DS at a fish company
    * We need to estimate the weight of fish using length, width and species
* Now again for those who took the intro class you'll knwo the answer
* This is the model
    * Show categorical unpooled model
* This is the fitted result
* But lets break down how we got here piece by piece

* In first section were going to talk about EDA
* In second section ..
* In next ...
* In final section you'll see how all this is just the Bayesian workflow in practice
* Lets get into the lesson!

## Section 20: Exploratory Data Analysis

* How do we know we can fit a linear model to a dataset?
    * Well we won't know if don't check
* This process of checking is called EDA
 Some of you may roll your eyes
    * I came here to learn Bayesian Statistics
    * EDA will help you be a better statistician.
* A relationship that is not well approximated by a straight line.
* A scatterplot that does not show a linear pattern.
    * You can describe it well with a straight line.
* The rate of change of the weight is not the same for all the fish lengths.
    * Too technical.
* The dependent variable does not decrease at a constant speed.
* We need to explore the data before proposing an appropriate model. Be wary of those who dare to write `.fit()` without doing EDA!
    * Load the data into Python.
    * Basic statistical summaries (`df.info()`, `df.summary()`, etc.)
    * It's always good to check associations using visualizations. 
        * Analyze relationship between size variables (scatterplots)
            * Non-linear outcomes!
            * What if we paint the dots with different colors based on species? Wow! This also gives us precious information!
        * Looks like relationship between length and width is linear
        * Looks like relationship between weight and length and width is not so linear 
    * Conclusions
      * We see some categories
      * We see some straight lines
      * We see some not so straight lines
* Now we have good understanding of our data. 
    * We're ready to start modeling!

### Section Recap

* We have to know what we're modeling
* EDA is absolutely essential
    * It's the way to go to get key information before start modeling.
    * Becomes more important when we get to later sections
    * Would you take the odds for a model that says bigger fish weigh less?
        * That's why EDA is important!
        * Also common sense
* Doing EDA right is what separates mediocre statisticians and DS from great ones
    * Trust us we've seen it enough times in real life
* Key insights
    * Longer fish weigh more.
    * We see non linear pattern.
        * We'll worry about that later
    * Fish of the same species tend to be grouped together.
        * Species is an important feature we don't want to leave aside in our model.

## Section 30: Intercept only regression

**Fun Title** The worlds simplest linear model

> Now it's the time. The moment we've been waiting for!

* We start as simple as possible.
    * `Y ~ 1`
    * The simplest straight line is flat.
* We're finally doing Bayesian Modeling!
* The model
    * Intuition
        * How does a flat line look like?
        * What does it mean?
    * Math
        * How do we represent a flat line using the equation of a straight line?
        * What does it mean?
    * PyMC code
        * Expressive and natural representation of the model
* Fit the model
* Analyze the results
    * Explore marginal posteriors
    * Explore traceplots
    * Explore fitted straight line
        * Scatterplot
        * Overlay mean line
        * Overlay credibility bands / other plausible lines

### Section Recap

* Straight lines are powerful building blocks for statistical modeling.
* Parameters in the linear function have an interpretable meaning.
    * In this special case the parameter equivalent the average
    * They are important to understand what the model wants to tell.
    * Expertise comes with practice!

* Uncertainty quantification for free. Going Bayesian is great!
    * Even for as things as simple as the mean
* The simplest linear regression model we can build is a flat line.
    * It omits all the information from any available predictor.
    * It assigns the same weight for all the fish.
    * It is known as the intercept only model.
* We discovered the learned intercept is equal to the mean response.

## Section 40: Adding a slope

* Let's make things more complex one step at a time.
    * Add one covariate
    * `Weight ~ 1 + Length`
    * Its called a covariate because it shows how the output _varies_ with changes in the input
    * Out intercept does not do this
* From the EDA we already know the larger the fish, the more its weight.
* We want to incorporate the length of the fish into the model.
    * How can we do that?
    * Using linear regression! The fish length goes into what we are used to call 'x'.
* The model
    * Intuition
        * How does the line look like now that we added a predictor?
        * What does it mean?
    * Math
        * We just only need to map the equation of a straight line to our problem.
    * PyMC Code
        * It's again expressive and natural.
        * Even more, we can extend from the previous model!
* Model fit
    * See there's nothing special. We can use the same code than before.
* Analyze results
    * Posteriors and traceplots look reasonable don't show any pathology.
* Predictions
    * Look how bad the fit is
    * Weight is non linear with length

### Section Recap

* Simple linear regression is powerful.
    * We could incorporate predictors into our predictive model.
    * All of this with proper uncertainty quantification.
* Linear fit was terrible in this case
* EDA is important to ensure the linear approximation applies to the problem at hand.
    * We knew to expect that from our EDA

## Section 50: Transformations

* `np.log(Weight) ~ 1 + np.log(Length)`
* What can we do while using a linear model
* We can transform
    * The response
    * The covariate
    * Or both
* When we do this and plot the data see how linear it logs in log space
    * The idea of transformation space is critical to the course
    * You're going to see it many many times
* Lets refit our model
    * Make predictions
    * Predictions are in logspace though. 
        * Is it a problem? No!
        * We can always transform back
        * This is your superpower as a mathematician
* Lets look at predictions
    * For individual of species of fish they still suck
    * Is reasonable that one intercept and one slope will fit all the fish in the world?
        * Would you predict the same weight?
        * Not accounting for species
  * _Insert picture of each one of the fish_
  
### Section Recap

* Model building is iterative
* Transformations make linear regression even more powerful.
    * They can make linear regression applicable in cases where the relationship of the natural variables is not linear.
    * It's not the end of the world if we don't observe a linear pattern at first sight.     
* Knowing the flexibility of transformations empowers us as data scientists


## Section 60: Accounting for the species

* `np.log(Weight) ~ 0 + species + species:np.log(Length)`

* We saw the species encodes valuable information to predict fish weight.
    * How can we add it to our existing model?
    * Well, the truth is PyMC makes it really easy for us. 
    * But we're not here to copy and paste code, but to understand what's going on under the hood.
* The first challenge is that species is a feature of different nature.
    * Fish length is numeric.
        * Not because it uses numbers for its values.
        * But because operations with those numbers are meaningful (we can add up two lengths, compute the mean, etc.)
    * Fish species is not numeric. It's categorical.
        * Not because it does not uses numbers for its values.
        * Even if we mapped species with numbers, math operations don't make sense!
* But hey, it's not such a big problem either!
    * Previously, we had a single group and a single regression line.
    * Now, we have multiple groups. Why don't we have multiple regression lines, one for each group?
* That turns out it's indeed the answer!
    * Adding a categorical predictor in a regression model is like splitting a single line into multiple lines, one per group.
    * Now each species can show a different behavior. We get a more flexible model!
* Wait, how can we make straight lines different?
    * Tweak the intercept: We end up with parallel lines
    * Tweak the slope: We end up with non-parallel lines that all share the same origin
    * Tweak both intercept and slope: We end up with non-parallel lines that are completely independent of each other.
    * These explanations are coupled with charts.
* In this section we move forward with the most flexible model (varying intercept and slopes model)
    * The other options can be included as an appendix.
* The model
    * Intuition is already covered before
    * Math
    * PyMC Code
* Model fit
* Analyze results

### Section Recap

* Regression models are extremely flexible.
    * We can incorporate predictors of different nature (numeric and categoric)
* Adding a categorical predictor is equivalent to splitting the regression line into multiple lines
    * We'll have as many lines as groups in the categorical predictor.
    * In our problem we have as many lines as species in the dataset.
* These lines are estimated independently for each species
    * ?Hierarchical modeling goes a step further and share information among groups.

## Section 70: Bayesian Workflow and Growing Pains

* `y ~ ...` TBD

* Along the lesson we saw simple straight models many times.
    * The numerical predictor (the length of the fish) is paired with a slope parameter.
    * At this point, we know very well how to interpret it.
* What if we wanted to add the height of the fish to the model as well? How do we do that?
    * More technically, how do we add another numerical predictor to the model? 
    * How does this impact the equation of the model?
* Turns out it's not complicated at all!
    * Multiple predictors means there are multiple slopes.
    * Each numerical predictor has an associated slope. 
    * If we have length and height we'll have a total of two slopes.
* How does it look like?
    * Math
    * Intuition: A plane
* How is it interpreted?
  * If we control for this variable we can see the effect of this other variable

### Section Recap

* The simple linear function can be extended to handle multiple predictors
    * Each numerical predictor is associated with one slope.
    * The more predictors the larger number of parameters in the model.
* The straight line evolves into a plane.
    * The plane is the natural extension of the straight line.
* The interpretation of the slope parameters is similar.
    * We just need to be a little more cautious.

## Exercises

* Do EDA of Penguin dataset
* Fit intercept only model
* Fit single covariate linear model
* Bring your own dataset
  * Fit a model and share it on Discourse


