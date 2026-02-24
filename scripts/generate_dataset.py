import json
import random

categories = ["HTML", "CSS", "JavaScript", "Node.js", "Express", "React", "MySQL"]

html_q = [
    "Explain the difference between semantic and non-semantic HTML elements. Provide examples.",
    "How do you include an external CSS file in HTML? Show the tag and attributes.",
    "What is the purpose of the <meta name=\"viewport\"> tag? Provide an example for responsive design.",
    "Describe how to create an accessible form control with a label linked to an input.",
    "How does the <picture> element improve responsive images? Give an example.",
]

css_q = [
    "Write a CSS rule to create a fixed header at the top of the page.",
    "How do you create a responsive grid that becomes a single column on small screens?",
    "Explain the difference between margin and padding with examples.",
    "Provide a CSS snippet to create a smooth hover transition on buttons.",
    "How do you use CSS variables (custom properties)? Give an example.",
]

js_q = [
    "Write a function to debounce another function in JavaScript.",
    "Explain event delegation and provide a short example.",
    "How do Promises differ from callbacks? Show a simple promise example.",
    "Write code to deep clone a simple object (no functions) in JavaScript.",
    "Explain the purpose of async/await with a fetch example.",
]

node_q = [
    "Write a basic Node.js script to read a file asynchronously and log its contents.",
    "Explain how require() differs from import in Node.js (CommonJS vs ESM).",
    "How do you handle uncaught exceptions in a Node.js process?",
    "Write a short example showing how to use environment variables in Node.js.",
    "Explain streams in Node.js and give an example of piping a readable stream to a writable stream.",
]

express_q = [
    "Create an Express route that accepts POST JSON and returns a 201 status.",
    "How would you structure middleware for logging requests in Express? Provide code.",
    "Explain how to serve static files in Express.",
    "Show how to set up error-handling middleware in Express.",
    "Write an example of route parameter extraction in Express (e.g., /users/:id).",
]

react_q = [
    "Write a simple functional component that displays a list of items passed as props.",
    "Explain useState and show an example counter component.",
    "How does useEffect differ when provided with [] vs [dep]? Provide examples.",
    "Describe lifting state up with a short example involving two child components.",
    "How do you memoize a component in React and why would you?",
]

mysql_q = [
    "Write a SQL query to find duplicate emails in a users table.",
    "Explain the difference between INNER JOIN and LEFT JOIN with examples.",
    "How do you create an index on a column and why would you do it?",
    "Write a query to paginate results using LIMIT and OFFSET.",
    "Explain transactions in MySQL and how to use BEGIN/COMMIT/ROLLBACK.",
]

question_pool = {
    "HTML": html_q,
    "CSS": css_q,
    "JavaScript": js_q,
    "Node.js": node_q,
    "Express": express_q,
    "React": react_q,
    "MySQL": mysql_q,
}


def make_example_answer(category, q):
    if category == "HTML":
        return "Explanation: " + q + " Example answer demonstrating correct usage."
    if category == "CSS":
        return "CSS snippet: /* example CSS */\n" + q
    if category == "JavaScript":
        return "JS Example: // example implementation for " + q
    if category == "Node.js":
        return "Node.js Example: // example implementation for " + q
    if category == "Express":
        return "Express Example: // example implementation for " + q
    if category == "React":
        return "React Example: // example implementation for " + q
    if category == "MySQL":
        return "SQL Example: -- example query for " + q
    return "Example answer"


def gen_entry(idx, category):
    q_list = question_pool[category]
    q = random.choice(q_list)
    example = make_example_answer(category, q)
    criteria = [
        {"aspect": "Correctness", "weight": 0.6},
        {"aspect": "Clarity", "weight": 0.25},
        {"aspect": "Conciseness", "weight": 0.15}
    ]
    entry = {
        "category": category,
        "question": q,
        "example_answer": example,
        "evaluation_criteria": criteria,
        "correctness": "Model should check correctness, clarity, and concision."
    }
    return entry


def generate_batch(count=100):
    items = []
    for i in range(count):
        category = categories[i % len(categories)]
        items.append(gen_entry(i, category))
    return items

if __name__ == '__main__':
    batch = generate_batch(100)
    out_path = 'datasets/webdev-coding-interview-expanded.json'
    with open(out_path, 'w') as f:
        json.dump(batch, f, indent=2)
    print(f'Wrote {len(batch)} entries to {out_path}')
