var ID = Math.floor(Math.random() * (9999999999 - 1000000000)) + 1000000000;
var outfile = ID + ".csv"
var turn = 1;

function isEvaluationDone() {
    for (let i = 0; i <= turn; i++) {
        var tmp = 0;
        var form = document.getElementsByName(i);
        if (form.length > 0) {
            for (let j = 0; j < form.length; j++) {
                if (form[j].checked == true) {
                    tmp = 1;
                }
            }
            if (tmp == 0) {
                return false;
            }
        }
    }
    return true;
}

function createEvalRow(name) {
    var article = document.createElement("article");
    article.className = "media"

    var eval = document.createElement("from");
    eval.id = "evaluation" + name;

    var evalMessagePara = document.createElement("p");
    var evalMessage = document.createElement("strong");
    evalMessage.innerHTML = "前の文脈に照らして上の発話は自然だと思いますか？"

    evalMessagePara.appendChild(evalMessage);
    eval.appendChild(evalMessagePara);

    EvalList = ["そう思わない", "ややそう思わない", "どちらでもない", "ややそう思う", "そう思う"];

    for (let i = 1; i < 6; i++) {
        const input_ = document.createElement("input");
        input_.type = "radio";
        input_.name = name;
        input_.id = "r" + i + name;
        input_.value = EvalList[i - 1];

        const text = document.createTextNode("  " + i + ". " + EvalList[i - 1]);
        const Label = document.createElement("label");
        Label.htmlFor = input_.id;
        Label.appendChild(input_);
        Label.appendChild(text);

        eval.appendChild(Label);
    }
    eval.style = "display:none";
    article.appendChild(eval);
    return article;
}

function apperAllFrom() {
    for (let i = 0; i <= turn; i++) {
        var form = document.getElementById("evaluation" + i);
        if (form != null) {
            form.style = "display:block";
        }
    }
}

function createChatRow(agent, text) {
    var article = document.createElement("article");
    article.className = "media"
    var figure = document.createElement("figure");
    figure.className = "media" + (agent === "You" ? "-right" : agent === "Model" ? "-left" : agent === "System" ? "-left" : "");
    var span = document.createElement("span");
    span.className = "icon is-large";
    var icon = document.createElement("i");
    icon.className = "fas fas fa-2x" + (agent === "You" ? " fa-user " : agent === "Model" ? " fa-robot" : agent === "System" ? " fa-robot" : "");
    var media = document.createElement("div");
    media.className = "media-content" + (agent === "You" ? "-right" : agent === "Model" ? "-left" : agent === "System" ? "" : "");
    var content = document.createElement("div");
    content.className = "content";

    var para2 = document.createElement("p");
    var paraText = document.createTextNode(text);
    para2.className = "balloon1" + (agent === "You" ? "-right" : agent === "Model" ? "-left" : "")
    var para1 = document.createElement("p");
    var strong = document.createElement("strong");
    strong.innerHTML = (agent === "You" ? "あなた" : agent === "Model" ? "システム" : agent === "System" ? " システムアナウンス" : agent);
    var br = document.createElement("br");

    para1.appendChild(strong);
    para2.appendChild(paraText);
    content.appendChild(para1);
    content.appendChild(para2);
    media.appendChild(content);

    if (agent === "You") {
        article.appendChild(media);
        span.appendChild(icon);
        figure.appendChild(span);
        article.appendChild(figure);
    } else {
        span.appendChild(icon);
        figure.appendChild(span);
        if (agent !== "Instructions") {
            article.appendChild(figure);
        };
        article.appendChild(media);
    }

    return article;
}

var context = [{
    "Talker": "S",
    "Uttr": "こんにちは。よろしくお願いします。"
}];
var pahse = 0

function exportCSV() {
    var csvData = "";
    for (var i = 0; i < context.length; i++) {
        csvData += "" + context[i]["Talker"] + "," +
            context[i]["Uttr"];
        var form = document.getElementsByName(i);
        var score = 0;
        if (form.length > 0) {
            for (let j = 0; j < form.length; j++) {
                if (form[j].checked == true) {
                    score = j + 1;
                }
            }
        }
        csvData += "," + score + "\r\n";
    }

    const link = document.createElement("a");
    document.body.appendChild(link);
    link.style = "display:none";
    const blob = new Blob([csvData], { type: "octet/stream" });
    const url = window.URL.createObjectURL(blob);
    link.href = url;
    link.download = outfile;
    link.click();
    window.URL.revokeObjectURL(url);
    link.parentNode.removeChild(link);
}

var parDiv = document.getElementById("parent");
parDiv.append(createChatRow("Model", "本日はどうぞよろしくお願いします。"));
parDiv.scrollTo(0, parDiv.scrollHeight);


document.getElementById("interact").addEventListener("submit", function(event) {
    event.preventDefault()

    if (pahse == 0) {
        ntex = document.getElementById("userIn").value;
        var ncontext = {
            "Talker": "U",
            "Uttr": ntex
        };
        context.push(ncontext);

        document.getElementById('userIn').value = "";
        var parDiv = document.getElementById("parent");
        parDiv.append(createChatRow("You", ntex));
        parDiv.scrollTo(0, parDiv.scrollHeight);
        turn += 1;
        var send_info = { "data": context, "ID": ID, "count": 11 - turn };
        document.getElementById("interact").style.display = "none";
        fetch('/interact_one_res', {
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST',
            body: JSON.stringify(send_info)
        }).then(response => response.json()).then(data => {
            // Change info for Model response
            parDiv.append(createChatRow("Model", data.text));
            parDiv.append(createEvalRow(turn));
            parDiv.scrollTo(0, parDiv.scrollHeight);
            document.getElementById("interact").style.display = "block";
            context.push({ "Talker": "S", "Uttr": data.text });
            turn += 1;
            if (turn >= 15) {
                pahse = 1;
                parDiv.append(createChatRow("System", "これにて対話は終了です。下の「評価開始」ボタンを押すと上に発話の評価欄が現れますので，各発話の評価を行ってください。"));
                document.getElementById("respond").textContent = "評価開始";
                document.getElementById("userIn").remove();
                parDiv.scrollTo(0, parDiv.scrollHeight);
            };
        })
    }
});


document.getElementById("interact").addEventListener("submit", function(event) {
    event.preventDefault();
    if (pahse == 1) {
        pahse = 2;
        apperAllFrom();
        console.log("EVAL");
        var parDiv = document.getElementById("parent");
        parDiv.append(createChatRow("System", "システム発話の評価を行ってください。評価が終わりましたら下の「評価の終了」ボタンを押して対話のダウンロードを行ってください。"));
        document.getElementById("respond").textContent = "評価終了";
    }
});

document.getElementById("interact").addEventListener("submit", function(event) {
    event.preventDefault();
    if (pahse == 2) {
        if (isEvaluationDone()) {
            exportCSV();
        }
    }
});