STYLE_SHEET = "https://cdnjs.cloudflare.com/ajax/libs/bulma/0.7.4/css/bulma.css"
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.3.1/js/all.js"
WEB_HTML = """
<html>
    <link rel="stylesheet" href={} />
    <style>
    .balloon1-left {{
      position: relative;
      display: inline-block;
      margin: 0.3em 0 0.3em 15px;
      padding: 7px 10px;
      min-width: 120px;
      max-width: 100%;
      color: #555;
      font-size: 16px;
      background: lightgray;
      border-radius: 5px 5px 5px 5px;
    }}

    .balloon1-left:before {{
      content: "";
      position: absolute;
      top: 50%;
      left: -30px;
      margin-top: -15px;
      border: 15px solid transparent;
      border-right: 15px solid lightgray;
    }}

    .balloon1-left p {{
      margin: 0;
      padding: 0;
    }}

    .balloon1-right {{
      position: relative;
      display: inline-block;
      margin: 0.3em 15px 0.3em 0;
      padding: 7px 10px;
      min-width: 120px;
      max-width: 100%;
      color: #555;
      font-size: 16px;
      background: rgba(144,238,144,0.7);
      border-radius: 5px 5px 5px 5px;
    }}

    .balloon1-right:before {{
      content: "";
      position: absolute;
      top: 50%;
      left: 100%;
      margin-top: -15px;
      border: 15px solid transparent;
      border-left: 15px solid rgba(144,238,144,0.7);
    }}

    .balloon1-right p {{
      margin: 0;
      padding: 0;
    }}

    .media {{
    align-items: flex-end;
    }}
    </style>
    <script defer src={}></script>
    <head><title> Interactive Run </title></head>
    <body>
        <div class="columns" style="height: 100%">
            <div class="column is-three-fifths is-offset-one-fifth">
              <section class="hero is-info is-large has-background-light has-text-grey-dark" style="height: 100%">
                <div id="parent" class="hero-body" style="overflow: auto; height: calc(100% - 76px); padding-top: 1em; padding-bottom: 0;">
                    <article class="media">
                      <div class="media-content">
                        <div class="content">
                          <p>
                            <strong>インストラクション</strong>
                            <br>
                            対話のページになります．<br>
                            絵文字を入力しないでください．<br>
                            対話をやり直す場合はブラウザを更新してください．<br>
                            発話を入力してから，若干ラグがあります．<br>
                            ただ，5秒以上たっても返信がない場合は クラウドワークスにて連絡をください<br>
                            すべて終わったら対話をダウンロードするボタンが下に出ますので，クリックして，クラウドワークスのお仕事ページにて提出をお願いします．
                        </p>
                        </div>
                      </div>
                    </article>
                </div>
                <div class="hero-foot column is-three-fifths is-offset-one-fifth" style="height: 76px">
                  <form id = "interact">
                      <div class="field is-grouped">
                        <p class="control is-expanded">
                          <input class="input" type="text" id="userIn" placeholder="メッセージを入力してください">
                        </p>
                        <p class="control">
                          <button id="respond" type="submit" class="button has-text-white-ter has-background-grey-dark">
                            Submit
                          </button>
                        </p>
                      </div>
                  </form>
                </div>
              </section>
            </div>
        </div>
        <script>
            var ID = Math.floor(Math.random() * (9999999999 - 1000000000)) + 1000000000;
            var outfile = ID + ".csv"
            var turn = 1;
            function createChatRow(agent, text) {{
                var article = document.createElement("article");
                article.className = "media"
                var figure = document.createElement("figure");
                figure.className = "media-right";
                var span = document.createElement("span");
                span.className = "icon is-large";
                var icon = document.createElement("i");
                icon.className = "fas fas fa-2x" + (agent === "You" ? " fa-user " : agent === "Model" ? " fa-robot" :  agent === "System" ? " fa-robot" : "");
                var media = document.createElement("div");
                media.className = "media-content";
                var content = document.createElement("div");
                content.className = "content";

                var para2 = document.createElement("p");
                var paraText = document.createTextNode(text);
                para2.className = "balloon1-" + (agent === "You" ? "right" : agent === "Model" ? "left" : agent === "System" ? "left"  : "");

                var para1 = document.createElement("p");
                var strong = document.createElement("strong");
                strong.innerHTML = (agent === "You" ? "あなた" : agent === "Model" ? "システム" :  agent === "System" ? " システムアナウンス" : agent);
                var br = document.createElement("br");

                para1.appendChild(strong);
                para2.appendChild(paraText);
                content.appendChild(para1);
                content.appendChild(para2);
                media.appendChild(content);

                article.appendChild(media);
                span.appendChild(icon);
                figure.appendChild(span);

                if (agent !== "Instructions") {{
                    article.appendChild(figure);
                }};
                return article;
            }}
            var context = [
                {{"Talker": "S",
                 "Uttr":   "こんにちは。よろしくお願いします。"
                }}];
            function exportCSV() {{
              var csvData = "";
              for (var i = 0; i < context.length; i++) {{
                  csvData += "" + context[i]["Talker"] + ","
                      + context[i]["Uttr"] + "\\r\\n";
              }}

              const link = document.createElement("a");
              document.body.appendChild(link);
              link.style = "display:none";
              const blob = new Blob([csvData], {{ type: "octet/stream" }});
              const url = window.URL.createObjectURL(blob);
              link.href = url;
              link.download = outfile;
              link.click();
              window.URL.revokeObjectURL(url);
              link.parentNode.removeChild(link);
              }}
            var parDiv = document.getElementById("parent");
            parDiv.append(createChatRow("Model", "こんにちは。よろしくお願いします。"));
            parDiv.scrollTo(0, parDiv.scrollHeight);
            document.getElementById("interact").addEventListener("submit", function(event){{
                event.preventDefault()

                ntex = document.getElementById("userIn").value;
                var ncontext = {{
                  "Talker": "U",
                 "Uttr":   ntex
                }};
                context.push(ncontext);

                document.getElementById('userIn').value = "";
                var parDiv = document.getElementById("parent");
                parDiv.append(createChatRow("You", ntex));
                parDiv.scrollTo(0, parDiv.scrollHeight);
                var send_info = {{"data": context, "ID": ID}};
                turn += 1;
                document.getElementById("interact").style.display = "none";
                fetch('/interact_one_res', {{
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    method: 'POST',
                    body: JSON.stringify(send_info)
                }}).then(response=>response.json()).then(data=>{{
                    // Change info for Model response
                    parDiv.append(createChatRow("Model", data.text));
                    parDiv.scrollTo(0, parDiv.scrollHeight);
                    document.getElementById("interact").style.display = "block";
                    context.push({{"Talker": "S", "Uttr": data.text}});
                    turn += 1;
                    if (turn >= 10) {{
                        parDiv.append(createChatRow("System", "これにて対話は終了です。下のボタンより対話のダウンロードを行ってください。"));
                        document.getElementById("respond").setAttribute("type", "reset");
                        document.getElementById("respond").textContent = "対話のダウンロード";
                        document.getElementById("userIn").remove();
                        parDiv.scrollTo(0, parDiv.scrollHeight);
                    }};
                }})
            }});
            document.getElementById("interact").addEventListener("reset", function(event){{
                event.preventDefault();
                exportCSV();
            }});
        </script>
    </style>
    </body>
</html>
"""  # noqa: E501