const express = require('express');
const cors = require('cors');
const app = express();
const PORT = process.env.PORT || 8080;
app.use(cors());
app.use(express.json());

// Define routes and middleware
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

app.get('/', function(req,res){
  res.render('client/database/db_conn.js')
  });
  