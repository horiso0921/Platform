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
                window.alert("5つすべての項目における評価をお願いします");
                return false;
            }
        }
    }
    return true;
}

function createEvalRow(name, content) {
    var article = document.createElement("article");
    article.className = "media"

    var eval = document.createElement("from");
    eval.id = "evaluation" + name;

    var evalMessagePara = document.createElement("p");
    var evalMessage = document.createElement("strong");
    evalMessage.innerHTML = content;

    evalMessagePara.appendChild(evalMessage);
    eval.appendChild(evalMessagePara);

    evalList = ["そう思わない", "ややそう思わない", "どちらでもない", "ややそう思う", "そう思う"];

    for (let i = 1; i < 6; i++) {
        const input_ = document.createElement("input");
        input_.type = "radio";
        input_.name = name;
        input_.id = "r" + i + name;
        input_.value = evalList[i - 1];

        const text = document.createTextNode("  " + i + ". " + evalList[i - 1]);
        const Label = document.createElement("label");
        Label.htmlFor = input_.id;
        Label.appendChild(input_);
        Label.appendChild(text);

        eval.appendChild(Label);
    }
    eval.style = "display:block";
    article.appendChild(eval);
    return article;
}

function apperAllFrom() {
    contentList = ["システムの発話は自然でしたか？", "システムの発話は情報量がありましたか？", "システムの発話は友好的でしたか？", "システムは質問を適切なタイミングで行っていましたか？", "システムは質問に対する回答を適切に行っていましたか？"];
    for (let i = 0; i < contentList.length; i++) {
        var parDiv = document.getElementById("parent");
        var j = i + 1;
        parDiv.append(createEvalRow(i, j + " : " + contentList[i]));
        console.log(i);
    }
}

function createChatRow(agent, text) {
    var article = document.createElement("article");
    article.className = "media"
    var figure = document.createElement("figure");
    var span = document.createElement("span");
    var icon = document.createElement("i");
    if (agent != "System") {
        figure.className = "media" + (agent === "You" ? "-right" : agent === "Model" ? "-left" : agent === "System" ? "-center" : "");
        span.className = "icon is-large";
        icon.className = "fas fas fa-2x" + (agent === "You" ? " fa-user " : agent === "Model" ? " fa-robot" : agent === "System" ? " fa-info-circle" : "");
    }
    var media = document.createElement("div");
    media.className = "media-content" + (agent === "You" ? "-right" : agent === "Model" ? "-left" : agent === "System" ? "-center" : "");
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
        if (agent != "System") {
            span.appendChild(icon);
            figure.appendChild(span);
        }
        if (agent !== "Instructions") {
            article.appendChild(figure);
        };
        article.appendChild(media);
    }

    return article;
}

var context = [];
var phase = 0

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
parDiv.append(createChatRow("System", "あなたが挨拶を入力すると対話がスタートします"));
parDiv.scrollTo(0, parDiv.scrollHeight);

document.getElementById("interact").addEventListener("submit", function (event) {
    event.preventDefault()

    if (phase == 0) {
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
        fetch('/interact', {
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST',
            body: JSON.stringify(send_info)
        }).then(response => response.json()).then(data => {
            // Change info for Model response
            parDiv.append(createChatRow("Model", data.text));
            parDiv.scrollTo(0, parDiv.scrollHeight);
            document.getElementById("interact").style.display = "block";
            context.push({ "Talker": "S", "Uttr": data.text });
            turn += 1;
            if (turn >= 15) {
                phase = 1;
                var end_phrase = "これにて対話は終了です。対話IDは " + ID + " です。"
                parDiv.append(createChatRow("System", end_phrase));
                parDiv.append(createChatRow("System", "下の「評価開始」ボタンを押すと，対話の評価欄が出現します。"));
                document.getElementById("respond").textContent = "評価開始";
                document.getElementById("userIn").remove();
                parDiv.scrollTo(0, parDiv.scrollHeight);
            };
        })
    }
});


document.getElementById("interact").addEventListener("submit", function (event) {
    event.preventDefault();
    if (phase == 1) {
        phase = 2;
        apperAllFrom();
        console.log("EVAL");
        var parDiv = document.getElementById("parent");
        parDiv.append(createChatRow("System", "評価が終わりましたら下の「評価の終了」ボタンを押して，やり取りのダウンロードを行ってください。"));
        document.getElementById("respond").textContent = "評価の終了";
        parDiv.scrollTo({
            top: parDiv.scrollHeight,
            behavior: 'smooth'
        });
    }
});

document.getElementById("interact").addEventListener("submit", function (event) {
    event.preventDefault();
    if (phase == 2) {
        if (isEvaluationDone()) {
            exportCSV();
        }
    }
});