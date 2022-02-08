const multer = require('multer')
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, "./images");
    },
    filename: function (req, file, cb) {
        cb(null, Date.now() + "-" + file.originalname.split(" ").join(""));
    },
})
const upload = multer({ storage })
module.exports = {
    upload,
}