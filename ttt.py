import gradio as gr

# Initial HTML content for the gr.HTML output
initial_html_content = """
<textarea id="output-textarea" rows="5" style="width:100%; box-sizing:border-box;"></textarea>
"""

def welcome(name):
    global initial_html_content  # Access the initial HTML content defined outside the function
    # Dictionary to map each speaker to a unique class name and color
    speaker_styles = {
        "SPEAKER_01": {"class": "speaker-01", "color": "green"},
        "SPEAKER_02": {"class": "speaker-02", "color": "blue"},
        "SPEAKER_03": {"class": "speaker-03", "color": "red"},
        "SPEAKER_04": {"class": "speaker-04", "color": "orange"}
    }
    
    # Process the input text
    html = name
    
    # Replace occurrences of each speaker with a span tag with a specific class
    for speaker, style in speaker_styles.items():
        if speaker in html:
            html = html.replace(speaker, f"<span class='{style['class']}'>{speaker}</span>")
    
    # Restore the initial HTML content
    html += initial_html_content
    return html

js = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';
    container.style.color = 'blue';  // Change color to blue

    var text = 'Welcome to Gradio!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 250);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""

# CSS styles for the speaker classes and the text area
css = """
.speaker-01 {
    color: green;
    font-weight: bold;
    font-style: italic;
}

.speaker-02 {
    color: blue;
    font-weight: bold;
    font-style: italic;
}

.speaker-03 {
    color: red;
    font-weight: bold;
    font-style: italic;
}

.speaker-04 {
    color: orange;
    font-weight: bold;
    font-style: italic;
}

/* Style for the text area */
#html-id {
    border: 2px solid black; /* Add border */
    padding: 10px; /* Add padding */
    width: 100%; /* Set width to 100% */
    box-sizing: border-box; /* Include padding and border in width */
}
"""

with gr.Blocks(js=js, css=css) as demo:
    inp = gr.Textbox(placeholder="What is your sentence?")
    out = gr.HTML(initial=initial_html_content, elem_id="html-id", elem_classes="html-css")
    inp.change(welcome, inp, out)

demo.launch()
