

// A list of all the data category names copied from the python script
const categories = ['Accommodation type (5 categories)', 'Adults and children in household (11 categories)', 'Car or van availability (3 categories)', 'Accommodation by type of dwelling (9 categories)', 'Combination of ethnic groups in household (8 categories)', 'Combination of religions in household (15 categories)', 'Dependent children in household and their age - indicator (3 categories)', 'Household Reference Person previously served in UK armed forces (5 categories)', 'Household deprivation (6 categories)', 'Household deprived in the education dimension (3 categories)', 'Household deprived in the employment dimension (3 categories)', 'Household deprived in the health and disability dimension (3 categories)', 'Household deprived in the housing dimension (3 categories)', 'Household language (English and Welsh) (5 categories)', 'Household size (5 categories)', 'Household type (6 categories)', 'Households with students or schoolchildren living away during term-time (4 categories)', 'Lifestage of Household Reference Person(13 categories)', 'Multiple ethnic groups in household (6 categories)', 'Multiple main languages in household (3 categories)', 'Number of Bedrooms (5 categories)', 'Number of adults in employment in household (5 categories)', 'Number of adults in household (3 categories)', 'Number of disabled adults in household (4 categories)', 'Number of disabled people in household (4 categories)', 'Number of disabled people in household whose day-to-day activities are limited a little (4 categories)', 'Number of disabled people in household whose day-to-day activities are limited a lot (4 categories)', 'Number of families in household (7 categories)', 'Number of people in household who previously served in UK armed forces (3 categories)', 'Number of people in household with a long-term heath condition but are not disabled (4 categories)', 'Number of people in household with no long-term health condition (4 categories)', 'Number of people per bedroom in household (5 categories)', 'Number of people per room in household (5 categories)', 'Number of people who work in household and their transport to work (18 categories)', 'Number of rooms (Valuation Office Agency) (6 categories)', 'Number of unpaid carers in household (6 categories)', 'Occupancy rating for bedrooms (5 categories)', 'Occupancy rating for rooms (5 categories)', 'Tenure of household (7 categories)', 'Type of central heating in household (13 categories)']


// reference 
// Code edited from stack overflow answer 
// Quentin https://stackoverflow.com/questions/866239/creating-the-checkbox-dynamically-using-javascript
// 2013, Stack Overflow, "Creating the checkbox dynamically using JavaScript?"
function populateDataForm() {
    // locates the html elements where the data categories and button will be stored
    var form = document.getElementById("dataCatergory")
    var container = document.getElementById("buttonContainer")
    var counter = 1
    // iterates over the categories list to create a checkbox item for each category
    // appends it to the form variable which is a html element
    categories.forEach(category => {
       var checkbox = document.createElement('input')
       checkbox.type = "checkbox"
       checkbox.id = counter
       checkbox.name = "categories"
       checkbox.className = "me-2"
       checkbox.value = category
       
       var label = document.createElement('label')
       label.for = counter
       label.innerHTML = category

       form.appendChild(checkbox)
       form.appendChild(label)
       form.appendChild(document.createElement('br'))
       counter = counter + 1
    })
    // Inserts a button into the html container element
    // This button submits the html form
        button = document.createElement('input')
        button.type = "submit"
        button.name = "submit"
        button.value = "Start Clustering"
        button.className = "btn btn-danger"
        container.appendChild(button)
}

populateDataForm()

