$(document).ready(function(){
    var sliders = ["#threshhold", "#maxConnections"];
    for (var i in sliders){
        var sldr = $(sliders[i]);
        sldr.next().html(sldr.val());
    }
    for (var i in sliders){
        var sldr = $(sliders[i]);
        sldr.on('input', function(){
            $(this).next().html($(this).val());
        });
    }

});

function changeParameters(){
    var sliders = ["#threshhold", "#maxConnections"];
    var params = [];
    for (var i in sliders){
        var sldr = $(sliders[i]);
        params.push(sldr.val());
    }
    var url_string = '/juggad?&threshhold_value='+params[0]+
                     '&max_connections_value='+params[1]
    ReLoadFile(encodeURI(url_string));
}

function ReLoadFile(url_string) {

    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {

            jsontext = this.response.split('|split|')[0];
            orignalText = this.response.split('|split|')[1];
            responseData=JSON.parse(jsontext);
            cy.elements().remove();
            load_graph();
        };
    }
        xhttp.open("GET", url_string, true);
        // xhttp.open("GET", "static/js/new_tree.json", true);
        xhttp.send();
}

function reset(){
    var sliders = ["#threshhold", "#maxConnections"];
    var default_values = [0.5, 5];
    for (var i in sliders){
        var sldr = $(sliders[i]);
        sldr.val(default_values[i]);
    }
    for (var i in sliders){
        var sldr = $(sliders[i]);
        sldr.next().html(sldr.val());
    }
    for (var i in sliders){
        var sldr = $(sliders[i]);
        sldr.on('input', function(){
            $(this).next().html($(this).val());
        });
    }
    changeParameters();
    load_graph();
}


function get_wiki_text(title){
    var url_string = '/wiki_text';
    var result;
    $.ajax({
        type: "GET",
        url: url_string,
        data: {url: encodeURI(title)},
        success: function(data) {
            console.log(data);
            result = data.responseText;
            return result;
        }
    });
    return result;
}
