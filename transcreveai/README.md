# AUIA

Aplicação web gratuita para transformar áudio e vídeo em texto diretamente no navegador. Não há login, banco de dados, API paga ou upload do arquivo do usuário.

## Como funciona

O AUIA usa [Transformers.js](https://github.com/huggingface/transformers.js), uma biblioteca open source que executa modelos ONNX no navegador com ONNX Runtime Web. O modelo multilíngue `onnx-community/whisper-small` é carregado em formato quantizado. Ele é mais pesado que o base, mas melhora a precisão em falas baixas, distantes e com ruído. A decodificação usa busca por cinco hipóteses (`num_beams`) para escolher palavras com mais contexto e reduzir erros de reconhecimento:

- **WebGPU + q4**, quando o navegador oferece WebGPU, para aproveitar a GPU;
- **WASM + q8**, como fallback gratuito para navegadores sem WebGPU.

O arquivo é decodificado e reamostrado localmente pela Web Audio API. Quando o navegador não consegue abrir um MP4, como acontece com alguns áudios exportados pelo WhatsApp, o AUIA usa `ffmpeg.wasm` como fallback local para extrair a trilha em WAV PCM. Somente a biblioteca, o FFmpeg e os pesos do modelo são baixados do CDN/Hugging Face na primeira execução; o áudio ou vídeo selecionado não é enviado para servidor. O modelo fica em cache no navegador.

A disponibilidade e velocidade variam conforme navegador, memória, CPU/GPU e codec instalado. WebGPU ainda é experimental em alguns navegadores. Arquivos muito longos ou aparelhos com pouca memória podem falhar; nesse caso, tente WAV/MP3 menor ou use um navegador Chromium atualizado.

O fluxo de conferência segue uma prática comum em editores de transcrição, como player com timestamps, lista de falas clicáveis e destaque do trecho em reprodução. A referência é apenas de fluxo de uso: o AUIA não usa a API do RecCloud, não copia código proprietário e não envia os arquivos para um serviço externo.

### Aprendizado local

Depois da transcrição, cada segmento pode ser ouvido pelo player e editado no campo ao lado. Ao clicar em **Salvar correções**, o AUIA guarda a relação entre o termo reconhecido e o termo corrigido no `localStorage`, separado por idioma. Nas próximas transcrições, essas correções são aplicadas automaticamente antes da exibição e da exportação. Esse aprendizado fica somente neste navegador e não altera os pesos do Whisper; para treinar um modelo de reconhecimento de fala seria necessário um processo de treinamento separado, com dados e recursos de máquina maiores.

Quando Whisper retorna segmentos com timestamps, o botão SRT usa esses timestamps. Caso o modelo não os forneça, o app cria um bloco SRT simples para evitar uma exportação vazia.

## Executar no Visual Studio Code

Não é necessário instalar dependências npm: a aplicação usa HTML, CSS, JavaScript e uma importação ESM do CDN.

No terminal do VS Code, entre na pasta do projeto:

```powershell
cd ".\transcreveai"
```

Inicie um servidor local. Uma opção usando apenas o Python já instalado é:

```powershell
python -m http.server 5500
```

Abra no navegador:

```text
http://localhost:5500
```

Também é possível usar a extensão **Live Server** do VS Code e abrir `index.html` com **Open with Live Server**.

## Primeiro uso

1. Selecione ou arraste um arquivo compatível.
2. O app começa em **Português (Brasil)**, que é o idioma mais confiável para este projeto. Se souber o idioma real, selecione-o para ganhar precisão. A opção automática é experimental porque o backend ONNX usado no navegador pode cair em inglês quando nenhum idioma é informado.
3. Clique em **Iniciar transcrição**.
4. Aguarde o download inicial do modelo e o processamento local.
5. Edite o texto e copie ou baixe TXT/SRT.

## Estrutura

```text
transcreveai/
├── index.html
├── style.css
├── script.js
├── README.md
└── assets/
```

A pasta `assets/` está reservada para futuros recursos visuais locais. A interface atual não depende de imagens externas.

## Limitações conhecidas

- O primeiro carregamento precisa de internet para obter Transformers.js e o modelo. Depois, o cache do navegador pode permitir reutilização offline, dependendo das políticas de cache.
- MP4s, MOVs, M4As e FLACs são processados pelo navegador quando os codecs estão disponíveis. Para arquivos como o MP4 do WhatsApp, o fallback `ffmpeg.wasm` extrai o áudio localmente e evita o erro de decodificação. A primeira conversão pode demorar e baixar cerca de 31 MB de runtime.
- A transcrição local exige mais recursos que um serviço remoto. O modelo small melhora a qualidade, mas aumenta bastante o download e o uso de memória. Em aparelhos fracos, tente um arquivo menor. Antes do Whisper, o áudio recebe filtro passa-altas, compressor de dinâmica e normalização local para recuperar fala baixa sem enviar o arquivo para servidor. O app também reduz repetições consecutivas causadas por janelas sobrepostas, sem alterar repetições naturais mais longas.
