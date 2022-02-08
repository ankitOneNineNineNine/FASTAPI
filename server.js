const express = require('express');
const cors = require('cors')
let app = express();
let axios = require('axios');
const { upload } = require('./upload');
app.use(express.json())
app.use(express.urlencoded({ extended: true }))
app.use(cors())

app.post('/predict', async function (req, res) {
    let text = req.body.text;
    try {
        let result = await axios.post(`http://127.0.0.1:8000/predict?line=${text}`);
        res.json(result.data)

    }
    catch (e) {
        res.json({
            'error': e
        })
    }
})

app.post('/imgPredict',upload.single('images'),  (req,res)=>{
    console.log(req.file)
    res.json('Done')
})


app.listen(3000, function () {
    console.log('listening to port')
})