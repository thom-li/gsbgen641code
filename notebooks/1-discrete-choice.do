* load Yogurt dataset
clear
use "/Users/thomasli/Library/CloudStorage/OneDrive-Stanford/*Y1 GSBGEN 641/gsbgen641code/data/YogurtLong.dta"

* add variables for outside option
rename nopurch b0
gen p0 = 0
gen f0 = 0

* create a unique identifier for each choice occasion (trip)
egen choice_id = group(hh tripnum)

* reshape data from wide to long
reshape long b p f, i(choice_id) j(brand)

* declare the data as choice model data
* 'choice_id' identifies the trip, 'brand' identifies the alternatives
cmset choice_id brand

* estimate the Conditional Logit Model
* 'b' is the dependent variable (1 if chosen, 0 otherwise)
* 'p' (price) and 'f' (feature) are alternative-specific independent variables
cmclogit b p f
